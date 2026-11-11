"""
Preprocessing stages -- replace the interactive Preprocessing menu.

``preprocessing.collect``    viral contigs of each tool -> the genome registry (+ renaming metadata)
``preprocessing.metadata``   class labels and scraped statistics -> the metadata CSV
``preprocessing.masks``      CheckV quality summaries -> MQ/HQ vMAG/provMAG masks
``preprocessing.visualize``  plots of the metadata CSV
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from phantom.runner.params import param
from phantom.runner.stage import RunContext, Stage, register
from phantom.stages._common import metadata_path, output_dir

IDENTIFIERS = ("genomad", "vibrant", "virsorter2")
Target = Literal["labeling", "runtimes", "checkv", "sizes"]
Plot = Literal["checkv_quality", "runtimes", "size_vs_runtime"]
PLOTS_DIR = ("plots", "metadata")


def _meta(ctx: RunContext):
    from phantom.preprocessing.metadata.mapping import MetadataManager

    return MetadataManager(MetadataManager.path_from_config(ctx.config))


def _require_paths(ctx: RunContext, sections: tuple[str, ...]) -> list[str]:
    return [
        f"config.toml is missing [{s}].path"
        for s in sections
        if not ctx.config.get(s, {}).get("path")
    ]


# --- preprocessing.collect -----------------------------------------------------


@dataclass(frozen=True)
class CollectParams:
    auto_format: bool = param(
        False, help="rename and include tool folders not named S<n>_<abbr> (else they are skipped)"
    )


@register
class Collect(Stage):
    name = "preprocessing.collect"
    summary = "Copy each identification tool's viral contigs into the genome registry"
    Params = CollectParams

    def validate(self, p: CollectParams, ctx: RunContext) -> list[str]:
        errors = _require_paths(ctx, ("genomes", *IDENTIFIERS))
        if not ctx.config.get("genomes", {}).get("metadata_path"):
            errors.append("config.toml is missing [genomes].metadata_path")
        return errors

    def check_inputs(self, p: CollectParams, ctx: RunContext) -> list[str]:
        missing = [
            str(ctx.section_path(t)) for t in IDENTIFIERS if not ctx.section_path(t).is_dir()
        ]
        return (
            [f"no identification-tool output in {', '.join(missing)}"] if len(missing) == 3 else []
        )

    def inputs(self, p: CollectParams, ctx: RunContext) -> list[Path]:
        return [ctx.section_path(t) for t in IDENTIFIERS]

    def writes(self, p: CollectParams, ctx: RunContext) -> list[Path]:
        return [ctx.section_path("genomes")]

    def run(self, p: CollectParams, ctx: RunContext) -> list[Path]:
        from phantom.preprocessing.metadata.gather import collect_genomes

        df = collect_genomes(ctx.config, _meta(ctx), auto_format=p.auto_format)
        print(f"[INFO] Collected genomes for {len(df)} samples.")
        return [ctx.section_path("genomes")]


# --- preprocessing.metadata ----------------------------------------------------


@dataclass(frozen=True)
class MetadataParams:
    targets: list[Target] = param(
        default_factory=lambda: ["labeling", "runtimes", "checkv", "sizes"],
        help="column groups to (re)compute",
    )
    labeling: dict[str, Any] | None = param(
        None,
        help='needed for "labeling": { strategy = "threshold", threshold = 34 } | '
        '{ strategy = "manual", positives = ["S1"] } | { strategy = "pattern", pattern = "..." } | '
        '{ strategy = "csv", path = "...", join_key = "id", label_col = "label" }',
    )
    overwrite_labels: bool = param(False, help="replace labels the metadata already has")


@register
class Metadata(Stage):
    name = "preprocessing.metadata"
    summary = "Class labels + runtimes / CheckV counts / assembly sizes into the metadata CSV"
    Params = MetadataParams

    def validate(self, p: MetadataParams, ctx: RunContext) -> list[str]:
        from phantom.preprocessing.metadata.gather import check_labeling_spec

        errors = []
        if not p.targets:
            errors.append("targets: give at least one of labeling, runtimes, checkv, sizes")
        if "labeling" in p.targets:
            if p.labeling is None:
                errors.append('targets include "labeling": give labeling = { strategy = ... }')
            else:
                errors += check_labeling_spec(p.labeling)
        elif p.labeling is not None:
            errors.append('labeling is given but "labeling" is not in targets')
        if not ctx.config.get("genomes", {}).get("metadata_path"):
            errors.append("config.toml is missing [genomes].metadata_path")
        return errors

    def check_inputs(self, p: MetadataParams, ctx: RunContext) -> list[str]:
        meta = _meta(ctx)
        if not meta.path.is_file():
            return [f"{meta.path} does not exist -- run preprocessing.collect first"]
        errors = []
        if "labeling" in p.targets and "label" in meta.load().columns and not p.overwrite_labels:
            errors.append(
                f"{meta.path} already has labels; set overwrite_labels = true to replace them"
            )
        if p.labeling and p.labeling.get("strategy") == "csv":
            path = ctx.resolve(p.labeling.get("path", ""))
            if not path.is_file():
                errors.append(f"labeling CSV not found: {path}")
        return errors

    def label(self, p: MetadataParams) -> str:
        return ", ".join(p.targets)

    def inputs(self, p: MetadataParams, ctx: RunContext) -> list[Path]:
        # The collected sample list, the tool outputs the statistics are read from.
        dirs = [ctx.section_path("genomes") / t for t in IDENTIFIERS]
        if {"runtimes", "checkv"} & set(p.targets):
            dirs += [ctx.section_path(t) for t in IDENTIFIERS]
        if "checkv" in p.targets:
            dirs.append(ctx.section_path("checkv"))
        if "sizes" in p.targets:
            dirs.append(ctx.section_path("megahit"))
        if p.labeling and p.labeling.get("strategy") == "csv":
            dirs.append(ctx.resolve(p.labeling["path"]))
        return dirs

    def writes(self, p: MetadataParams, ctx: RunContext) -> list[Path]:
        return [metadata_path(ctx)]

    def run(self, p: MetadataParams, ctx: RunContext) -> list[Path]:
        from phantom.preprocessing.metadata.gather import build_labeler, gather_metadata

        labeler = build_labeler(p.labeling, ctx.resolve) if p.labeling else None
        columns = gather_metadata(
            ctx.config, _meta(ctx), list(p.targets), labeler, overwrite_labels=p.overwrite_labels
        )
        print(f"[INFO] Wrote {len(columns)} column(s) to {metadata_path(ctx)}")
        return []  # a file, not a directory: nothing to put a run.toml into


# --- preprocessing.masks -------------------------------------------------------


@dataclass(frozen=True)
class MasksParams:
    medium_completeness: float = param(50.0, help="MQ masks: completeness >= this (%)")
    high_completeness: float = param(75.0, help="HQ masks: completeness >= this (%)")
    max_contamination: float = param(10.0, help="every mask: contamination < this (%)")


@register
class Masks(Stage):
    name = "preprocessing.masks"
    summary = "CheckV quality summaries -> MQ/HQ vMAG and provMAG masks per tool"
    Params = MasksParams

    def validate(self, p: MasksParams, ctx: RunContext) -> list[str]:
        errors = _require_paths(ctx, ("checkv", "masks"))
        if not 0 <= p.medium_completeness <= p.high_completeness <= 100:
            errors.append("need 0 <= medium_completeness <= high_completeness <= 100")
        if not 0 < p.max_contamination <= 100:
            errors.append("max_contamination must be in (0, 100]")
        return errors

    def check_inputs(self, p: MasksParams, ctx: RunContext) -> list[str]:
        checkv = ctx.section_path("checkv")
        if not any(checkv.glob("*/*/quality_summary.tsv")):
            return [f"no CheckV quality_summary.tsv under {checkv} -- run tools.checkv first"]
        return []

    def label(self, p: MasksParams) -> str:
        return f"MQ>={p.medium_completeness:g} HQ>={p.high_completeness:g} cont<{p.max_contamination:g}"

    def inputs(self, p: MasksParams, ctx: RunContext) -> list[Path]:
        return [ctx.section_path("checkv")]

    def writes(self, p: MasksParams, ctx: RunContext) -> list[Path]:
        return [ctx.section_path("masks")]

    def run(self, p: MasksParams, ctx: RunContext) -> list[Path]:
        from phantom.preprocessing.masks.generator import CheckvMaskGenerator

        generator = CheckvMaskGenerator(
            ctx.config,
            medium_completeness=p.medium_completeness,
            high_completeness=p.high_completeness,
            max_contamination=p.max_contamination,
        )
        if not generator.run():
            raise RuntimeError("no CheckV results found to generate masks from")
        return generator.written


# --- preprocessing.visualize ---------------------------------------------------


@dataclass(frozen=True)
class VisualizeParams:
    plots: list[Plot] | None = param(None, help="default: every plot the metadata has columns for")
    out_dir: str | None = param(
        None, help="default: <[results]>/plots/metadata; relative = under the workspace"
    )


@register
class Visualize(Stage):
    name = "preprocessing.visualize"
    summary = "Plots of the metadata: CheckV quality, tool runtimes, assembly size vs. runtime"
    Params = VisualizeParams

    def check_inputs(self, p: VisualizeParams, ctx: RunContext) -> list[str]:
        from phantom.preprocessing.metadata.visualizer import MetadataVisualizer

        meta = _meta(ctx)
        if not meta.path.is_file():
            return [f"{meta.path} does not exist -- run preprocessing.collect first"]
        available = MetadataVisualizer.available_plots(meta.load())
        if not available:
            return ["the metadata has no columns to plot -- run preprocessing.metadata first"]
        return [
            f"plot {name!r} needs metadata columns that are missing (available: {', '.join(available)})"
            for name in p.plots or []
            if name not in available
        ]

    def label(self, p: VisualizeParams) -> str:
        return ", ".join(p.plots) if p.plots else "all available"

    def inputs(self, p: VisualizeParams, ctx: RunContext) -> list[Path]:
        return [metadata_path(ctx)]

    def writes(self, p: VisualizeParams, ctx: RunContext) -> list[Path]:
        return [output_dir(p.out_dir, ctx, *PLOTS_DIR)]

    def run(self, p: VisualizeParams, ctx: RunContext) -> list[Path]:
        from phantom.preprocessing.metadata.visualizer import MetadataVisualizer

        df = _meta(ctx).load()
        names: list[str] = list(p.plots) if p.plots else MetadataVisualizer.available_plots(df)
        out_dir = output_dir(p.out_dir, ctx, *PLOTS_DIR)
        for path in MetadataVisualizer.render(df, names, out_dir):
            print(f"[INFO] {path}")
        return [out_dir]
