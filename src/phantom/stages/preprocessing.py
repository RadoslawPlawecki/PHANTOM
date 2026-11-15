"""
Preprocessing stages: from the user's raw assemblies and the identification
tools' output to labeled samples.

``preprocessing.samples``    raw folder -> ids S1..Sn, S<n>.fa assembly links, labels, mapping
``preprocessing.collect``    viral contigs of each tool -> the genome registry (+ renaming metadata)
``preprocessing.metadata``   class labels and scraped statistics -> the metadata CSV
``preprocessing.masks``      CheckV quality summaries -> MQ/HQ vMAG/provMAG masks
``preprocessing.visualize``  plots of the metadata CSV
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from phantom.runner.parameters import param
from phantom.runner.stage import RunContext, Stage, register
from phantom.stages._common import metadata_path, output_dir

IDENTIFIERS = ("genomad", "vibrant", "virsorter2")
Target = Literal["labeling", "runtimes", "checkv", "sizes"]
Plot = Literal["checkv_quality", "runtimes", "size_vs_runtime"]
PLOTS_DIR = ("plots", "metadata")


def _metadata_table(context: RunContext):
    from phantom.analysis.genomes.metadata.table import MetadataTable

    return MetadataTable(MetadataTable.path_from_config(context.config))


def _require_paths(context: RunContext, sections: tuple[str, ...]) -> list[str]:
    return [
        f"config.toml is missing [{s}].path"
        for s in sections
        if not context.config.get(s, {}).get("path")
    ]


# preprocessing.samples


@dataclass(frozen=True)
class SamplesParams:
    pass


@register
class Samples(Stage):
    name = "preprocessing.samples"
    summary = (
        "Raw folder (one subfolder per class) -> ids S1..Sn, S<n>.fa links, the sample mapping"
    )
    Params = SamplesParams

    @staticmethod
    def _read(context: RunContext):
        from phantom.analysis.genomes.samples import read_samples

        return read_samples(
            context.section_path("raw"),
            context.config.get("classes", {}),
            _metadata_table(context).load(),
        )

    def validate(self, params: SamplesParams, context: RunContext) -> list[str]:
        from phantom.analysis.genomes.samples import class_folders

        errors = _require_paths(context, ("raw", "samples", "assemblies"))
        if not context.config.get("genomes", {}).get("metadata_path"):
            errors.append("config.toml is missing [genomes].metadata_path")
        return errors + class_folders(context.config.get("classes", {}))[1]

    def check_inputs(self, params: SamplesParams, context: RunContext) -> list[str]:
        return self._read(context)[1]

    def inputs(self, params: SamplesParams, context: RunContext) -> list[Path]:
        # The raw folder (fingerprinted file by file) and the files it links to.
        return [context.section_path("raw"), *(s.assembly for s in self._read(context)[0])]

    def writes(self, params: SamplesParams, context: RunContext) -> list[Path]:
        return [
            context.section_path("assemblies"),
            context.section_path("samples"),
            metadata_path(context),
        ]

    def run(self, params: SamplesParams, context: RunContext) -> list[Path]:
        import pandas as pd

        from phantom.analysis.genomes.samples import (
            ASSEMBLY,
            LABEL,
            link_assemblies,
            metadata_frame,
            write_mapping,
        )

        metadata = _metadata_table(context)
        samples, errors = self._read(context)
        if errors:
            raise ValueError("; ".join(errors))
        before = metadata.load()
        if LABEL in before.columns:
            labels = dict(zip(before["id"], before[LABEL], strict=True))
            for s in samples:
                if s.id in labels and pd.notna(labels[s.id]) and int(labels[s.id]) != s.label:
                    print(f"[INFO] {s.id} ({s.key}) relabeled {int(labels[s.id])} -> {s.label}")
        directory = context.section_path("assemblies")
        link_assemblies(samples, directory)
        mapping = context.section_path("samples")
        write_mapping(samples, mapping)
        metadata.add_metadata_columns(metadata_frame(samples), overwrite={ASSEMBLY, LABEL})
        cases = sum(s.label for s in samples)
        print(
            f"[INFO] {len(samples)} samples ({cases} cases, {len(samples) - cases} controls): "
            f"{samples[0].id}..{samples[-1].id} linked in {directory}; mapping in {mapping}"
        )
        return [directory]


# preprocessing.collect


@dataclass(frozen=True)
class CollectParams:
    pass


@register
class Collect(Stage):
    name = "preprocessing.collect"
    summary = "Copy each identification tool's viral contigs into the genome registry"
    Params = CollectParams

    def validate(self, params: CollectParams, context: RunContext) -> list[str]:
        errors = _require_paths(context, ("genomes", *IDENTIFIERS))
        if not context.config.get("genomes", {}).get("metadata_path"):
            errors.append("config.toml is missing [genomes].metadata_path")
        return errors

    def check_inputs(self, params: CollectParams, context: RunContext) -> list[str]:
        missing = [
            str(context.section_path(t))
            for t in IDENTIFIERS
            if not context.section_path(t).is_dir()
        ]
        return (
            [f"no identification tool output in {', '.join(missing)}"] if len(missing) == 3 else []
        )

    def inputs(self, params: CollectParams, context: RunContext) -> list[Path]:
        return [context.section_path(t) for t in IDENTIFIERS]

    def writes(self, params: CollectParams, context: RunContext) -> list[Path]:
        return [context.section_path("genomes")]

    def run(self, params: CollectParams, context: RunContext) -> list[Path]:
        from phantom.analysis.genomes.metadata.gather import collect_genomes

        df = collect_genomes(context.config, _metadata_table(context))
        print(f"[INFO] Collected genomes for {len(df)} samples.")
        return [context.section_path("genomes")]


# preprocessing.metadata


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

    def validate(self, params: MetadataParams, context: RunContext) -> list[str]:
        from phantom.analysis.genomes.metadata.gather import check_labeling_spec

        errors = []
        if not params.targets:
            errors.append("targets: give at least one of labeling, runtimes, checkv, sizes")
        if "labeling" in params.targets:
            if params.labeling is None:
                errors.append('targets include "labeling": give labeling = { strategy = ... }')
            else:
                errors += check_labeling_spec(params.labeling)
        elif params.labeling is not None:
            errors.append('labeling is given but "labeling" is not in targets')
        if not context.config.get("genomes", {}).get("metadata_path"):
            errors.append("config.toml is missing [genomes].metadata_path")
        if "checkv" in params.targets:
            errors += _require_paths(context, ("checkv", "masks"))
        # A labels file is the user's input, not a pipeline output: check it
        # up front, before hours of earlier steps run.
        if (
            params.labeling
            and params.labeling.get("strategy") == "csv"
            and params.labeling.get("path")
        ):
            path = context.resolve(params.labeling["path"])
            if not path.is_file():
                errors.append(f"labels file not found: {path} (columns: id,label)")
        return errors

    def check_inputs(self, params: MetadataParams, context: RunContext) -> list[str]:
        metadata = _metadata_table(context)
        if not metadata.path.is_file():
            return [f"{metadata.path} does not exist -- run preprocessing.collect first"]
        errors = []
        if (
            "labeling" in params.targets
            and "label" in metadata.load().columns
            and not params.overwrite_labels
        ):
            errors.append(
                f"{metadata.path} already has labels; set overwrite_labels = true to replace them"
            )
        if "checkv" in params.targets:
            masks = context.section_path("masks")
            if not any(masks.glob("*/MQ_provMAGs.tsv")):
                errors.append(
                    f"targets include checkv: no masks in {masks} -- run preprocessing.masks first "
                    "(the quality counts are read from them)"
                )
        return errors

    def label(self, params: MetadataParams) -> str:
        return ", ".join(params.targets)

    def inputs(self, params: MetadataParams, context: RunContext) -> list[Path]:
        # The collected sample list, the tool outputs the statistics are read from.
        dirs = [context.section_path("genomes") / t for t in IDENTIFIERS]
        if {"runtimes", "checkv"} & set(params.targets):
            dirs += [context.section_path(t) for t in IDENTIFIERS]
        if "checkv" in params.targets:
            dirs += [context.section_path("checkv"), context.section_path("masks")]
        if "sizes" in params.targets:
            dirs.append(context.section_path("assemblies"))
        if params.labeling and params.labeling.get("strategy") == "csv":
            dirs.append(context.resolve(params.labeling["path"]))
        return dirs

    def writes(self, params: MetadataParams, context: RunContext) -> list[Path]:
        return [metadata_path(context)]

    def run(self, params: MetadataParams, context: RunContext) -> list[Path]:
        from phantom.analysis.genomes.metadata.gather import build_labeler, gather_metadata

        labeler = build_labeler(params.labeling, context.resolve) if params.labeling else None
        columns = gather_metadata(
            context.config,
            _metadata_table(context),
            list(params.targets),
            labeler,
            overwrite_labels=params.overwrite_labels,
        )
        print(f"[INFO] Wrote {len(columns)} column(s) to {metadata_path(context)}")
        return []  # a file, not a directory: nothing to put a run.toml into


# preprocessing.masks


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

    def validate(self, params: MasksParams, context: RunContext) -> list[str]:
        errors = _require_paths(context, ("checkv", "masks"))
        if not 0 <= params.medium_completeness <= params.high_completeness <= 100:
            errors.append("need 0 <= medium_completeness <= high_completeness <= 100")
        if not 0 < params.max_contamination <= 100:
            errors.append("max_contamination must be in (0, 100]")
        return errors

    def check_inputs(self, params: MasksParams, context: RunContext) -> list[str]:
        checkv = context.section_path("checkv")
        if not any(checkv.glob("*/*/quality_summary.tsv")):
            return [f"no CheckV quality_summary.tsv under {checkv} -- run tools.checkv first"]
        return []

    def label(self, params: MasksParams) -> str:
        return f"MQ>={params.medium_completeness:g} HQ>={params.high_completeness:g} cont<{params.max_contamination:g}"

    def inputs(self, params: MasksParams, context: RunContext) -> list[Path]:
        return [context.section_path("checkv")]

    def writes(self, params: MasksParams, context: RunContext) -> list[Path]:
        return [context.section_path("masks")]

    def run(self, params: MasksParams, context: RunContext) -> list[Path]:
        from phantom.analysis.genomes.masks import CheckvMaskGenerator

        generator = CheckvMaskGenerator(
            context.config,
            medium_completeness=params.medium_completeness,
            high_completeness=params.high_completeness,
            max_contamination=params.max_contamination,
        )
        if not generator.run():
            raise RuntimeError("no CheckV results found to generate masks from")
        return generator.written


# preprocessing.visualize


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

    def check_inputs(self, params: VisualizeParams, context: RunContext) -> list[str]:
        from phantom.analysis.genomes.metadata.plots import MetadataPlots

        metadata = _metadata_table(context)
        if not metadata.path.is_file():
            return [f"{metadata.path} does not exist -- run preprocessing.collect first"]
        available = MetadataPlots.available_plots(metadata.load())
        if not available:
            return ["the metadata has no columns to plot -- run preprocessing.metadata first"]
        return [
            f"plot {name!r} needs metadata columns that are missing (available: {', '.join(available)})"
            for name in params.plots or []
            if name not in available
        ]

    def label(self, params: VisualizeParams) -> str:
        return ", ".join(params.plots) if params.plots else "all available"

    def inputs(self, params: VisualizeParams, context: RunContext) -> list[Path]:
        return [metadata_path(context)]

    def writes(self, params: VisualizeParams, context: RunContext) -> list[Path]:
        return [output_dir(params.out_dir, context, *PLOTS_DIR)]

    def run(self, params: VisualizeParams, context: RunContext) -> list[Path]:
        from phantom.analysis.genomes.metadata.plots import MetadataPlots

        df = _metadata_table(context).load()
        names: list[str] = list(params.plots) if params.plots else MetadataPlots.available_plots(df)
        out_dir = output_dir(params.out_dir, context, *PLOTS_DIR)
        for path in MetadataPlots.render(df, names, out_dir):
            print(f"[INFO] {path}")
        return [out_dir]
