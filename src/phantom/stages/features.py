"""
Feature-building stages (import -> merge -> preprocess -> extract).

``features.import``      copy external tool outputs into a version's raw/ stage
``features.merge``       concatenate per-sample raw files into raw-merged/
``features.preprocess``  per-tool cleaning (+ CheckV mask) into preprocessed/
``features.extract``     per-tool feature matrices into extracted/

Inputs/outputs are declared as stage directories (not discovered file
lists), so a run file can chain these steps on a version that does not exist
yet -- e.g. import (create) -> merge -> preprocess -> extract -> classify.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from phantom.runner.params import param
from phantom.runner.stage import RunContext, Stage, register
from phantom.stages._common import check_version

VERSION_HELP = "feature version, a [features.vX_Y] block in config.toml"
Tool = Literal["cherry", "phagcn", "phavip", "phatyp"]
TOOLS: tuple[str, ...] = ("cherry", "phagcn", "phavip", "phatyp")
MAX_MIN_PATIENTS = 10  # upper bound for min_patients


def _pipeline_flags(tool: str) -> tuple[bool, bool]:
    """(needs a feature column, uses binary/min_patients) for a tool."""
    from phantom.features.extraction.extractor import PIPELINES

    cls = PIPELINES[tool]
    return bool(getattr(cls, "NEEDS_FEATURE_COLUMN", False)), bool(
        getattr(cls, "NEEDS_MATRIX_OPTIONS", False)
    )


def _tool_dirs(
    ctx: RunContext, version: str, stage: str, tools: Sequence[str] | None
) -> list[Path]:
    """The stage dir itself (all tools) or one subdirectory per requested tool."""
    root = ctx.stage_dir(version, stage)
    return [root] if tools is None else [root / t for t in tools]


def _present_tools(
    ctx: RunContext, version: str, stage: str, tools: Sequence[str] | None
) -> list[str]:
    root = ctx.stage_dir(version, stage)
    wanted = TOOLS if tools is None else tools
    return [t for t in TOOLS if t in wanted and (root / t).is_dir()]


# --- features.import ----------------------------------------------------------


@dataclass(frozen=True)
class ImportParams:
    version: str = param(help='feature version to import into, e.g. "4.0"')
    sources: list[str] = param(
        help="files/directories to copy into <version>/raw/ (relative = under the workspace)"
    )
    create: bool = param(False, help="true = create the version (adds a block to config.toml)")
    overwrite: bool = param(False, help="replace a source already present in raw/")


@register
class Import(Stage):
    name = "features.import"
    summary = "Copy external tool outputs into a (new) feature version's raw stage"
    Params = ImportParams

    def announces(self, p: ImportParams) -> list[str]:
        return [p.version] if p.create else []

    def validate(self, p: ImportParams, ctx: RunContext) -> list[str]:
        errors = []
        if p.create and p.version in ctx.feature_versions():
            errors.append(f"version {p.version!r} already exists; use create = false to add to it")
        if not p.create:
            errors += check_version(p.version, ctx)
        if not p.sources:
            errors.append("sources: give at least one path")
        return errors

    def check_inputs(self, p: ImportParams, ctx: RunContext) -> list[str]:
        return [f"source not found: {s}" for s in p.sources if not ctx.resolve(s).exists()]

    def label(self, p: ImportParams) -> str:
        return f"{len(p.sources)} source(s) -> {p.version}"

    def inputs(self, p: ImportParams, ctx: RunContext) -> list[Path]:
        return [ctx.resolve(s) for s in p.sources]

    def writes(self, p: ImportParams, ctx: RunContext) -> list[Path]:
        return [ctx.stage_dir(p.version, "raw")]

    def run(self, p: ImportParams, ctx: RunContext) -> list[Path]:
        from phantom.features.collection.importer import FeatureImporter

        importer = FeatureImporter(config_path=ctx.config_path)
        raw_dir, failed = importer.import_sources(
            [ctx.resolve(s) for s in p.sources], p.version, create=p.create, overwrite=p.overwrite
        )
        if failed:
            raise RuntimeError("could not copy: " + "; ".join(failed))
        if p.create:
            ctx.reload()  # config.toml gained a [features.vX_Y] block
        return [raw_dir]


# --- features.merge -----------------------------------------------------------


@dataclass(frozen=True)
class MergeParams:
    version: str = param(help=VERSION_HELP)
    id_columns: dict[str, str] = param(
        help="sequence-ID column per raw/ subdirectory, first matching glob wins, e.g. "
        '{ "phavip/*" = "Genome", "*" = "Accession" }'
    )


@register
class Merge(Stage):
    name = "features.merge"
    summary = "Concatenate per-sample raw files into one merged file per tool/vtool"
    Params = MergeParams

    def choices(self, ctx: RunContext) -> dict[str, list]:
        return {"version": ctx.feature_versions()}

    def _concatenator(self, p: MergeParams, ctx: RunContext):
        from phantom.features.merging.concatenator import FeatureConcatenator

        return FeatureConcatenator(p.version, ctx.config_mgr, p.id_columns)

    def validate(self, p: MergeParams, ctx: RunContext) -> list[str]:
        errors = check_version(p.version, ctx)
        if not p.id_columns:
            errors.append('id_columns: give at least one pattern, e.g. { "*" = "Accession" }')
        return errors

    def check_inputs(self, p: MergeParams, ctx: RunContext) -> list[str]:
        concat = self._concatenator(p, ctx)
        leaves = concat.leaf_dirs()
        if not leaves:
            return [f"no .tsv/.csv files under {concat.raw_dir}"]
        errors = []
        for leaf in leaves:
            label = str(leaf.relative_to(concat.raw_dir))
            first = concat.raw_files(leaf)[0]
            columns = concat.peek_columns(first)
            column = concat.id_column_for(label)
            if column is None:
                errors.append(f"id_columns: no pattern matches raw/{label}")
            elif column not in columns:
                errors.append(
                    f"id_columns: {column!r} (for raw/{label}) is not a column of {first.name} "
                    f"(columns: {', '.join(columns)})"
                )
        return errors

    def label(self, p: MergeParams) -> str:
        return p.version

    def inputs(self, p: MergeParams, ctx: RunContext) -> list[Path]:
        return [ctx.stage_dir(p.version, "raw")]

    def writes(self, p: MergeParams, ctx: RunContext) -> list[Path]:
        return [ctx.stage_dir(p.version, "raw_merged")]

    def run(self, p: MergeParams, ctx: RunContext) -> list[Path]:
        written = self._concatenator(p, ctx).run()
        return sorted({f.parent for f in written})


# --- features.preprocess ------------------------------------------------------


@dataclass(frozen=True)
class PreprocessParams:
    version: str = param(help=VERSION_HELP)
    tools: list[Tool] | None = param(None, help="default: every tool present in raw-merged/")
    mask: str | None = param(
        None, help='CheckV mask file in <[masks]>/<vtool>/, e.g. "MQ_vMAGs.tsv"; default: none'
    )


@register
class Preprocess(Stage):
    name = "features.preprocess"
    summary = "Per-tool cleaning of merged files (optionally CheckV-masked)"
    Params = PreprocessParams

    def choices(self, ctx: RunContext) -> dict[str, list]:
        masks = ctx.section_path("masks")
        names = sorted({m.name for m in masks.glob("*/*.tsv")}) if masks.is_dir() else []
        return {"version": ctx.feature_versions(), "mask": names}

    def _mask_files(self, p: PreprocessParams, ctx: RunContext) -> list[Path]:
        """The mask file of every configured vtool (one per vtool directory)."""
        if not p.mask:
            return []
        root = ctx.section_path("masks")
        return [root / tool_name / p.mask for tool_name in ctx.config.get("tools", {}).values()]

    def validate(self, p: PreprocessParams, ctx: RunContext) -> list[str]:
        errors = check_version(p.version, ctx)
        if p.mask and not ctx.config.get("masks", {}).get("path"):
            errors.append("config.toml is missing [masks].path")
        return errors

    def check_inputs(self, p: PreprocessParams, ctx: RunContext) -> list[str]:
        from phantom.features.preprocessing.preprocessor import mask_dir_for

        root = ctx.stage_dir(p.version, "raw_merged")
        files = [
            f
            for t in _present_tools(ctx, p.version, "raw_merged", p.tools)
            for f in sorted((root / t).glob("*.csv"))
        ]
        if not files:
            wanted = ", ".join(p.tools) if p.tools else "any tool"
            return [f"no merged files for {wanted} under {root}"]
        errors = []
        for f in files:
            try:
                mask_dir = mask_dir_for(f, ctx.config)
            except ValueError as e:
                errors.append(str(e))
                continue
            if p.mask and not (mask_dir / p.mask).is_file():
                errors.append(f"mask not found: {mask_dir / p.mask}")
        return errors

    def label(self, p: PreprocessParams) -> str:
        return f"{p.version} / mask: {p.mask or 'none'}"

    def inputs(self, p: PreprocessParams, ctx: RunContext) -> list[Path]:
        return [*_tool_dirs(ctx, p.version, "raw_merged", p.tools), *self._mask_files(p, ctx)]

    def writes(self, p: PreprocessParams, ctx: RunContext) -> list[Path]:
        return _tool_dirs(ctx, p.version, "preprocessed", p.tools)

    def run(self, p: PreprocessParams, ctx: RunContext) -> list[Path]:
        from phantom.features.preprocessing.preprocessor import FeaturePreprocessor

        return FeaturePreprocessor(
            p.version,
            ctx.config,
            ctx.config_mgr,
            mask=p.mask,
            tools=_present_tools(ctx, p.version, "raw_merged", p.tools),
        ).run()


# --- features.extract ---------------------------------------------------------


@dataclass(frozen=True)
class ExtractParams:
    version: str = param(help=VERSION_HELP)
    tools: list[Tool] | None = param(None, help="default: every tool present in preprocessed/")
    feature_col: dict[str, str] = param(
        default_factory=dict,
        help="taxonomy column per tool that needs one, "
        'e.g. { cherry = "ncbi_genus", phagcn = "genus" }',
    )
    binary: bool | dict[str, bool] | None = param(
        None,
        help="presence/absence (true) or counts (false); one value or per tool; "
        "default: the pipeline's",
    )
    min_patients: int | dict[str, int] | None = param(
        None,
        help=f"keep features seen in >= N patients (1-{MAX_MIN_PATIENTS}); "
        "one value or per tool; default: the pipeline's",
    )


@register
class Extract(Stage):
    name = "features.extract"
    summary = "Build per-tool feature matrices from preprocessed data"
    Params = ExtractParams

    def choices(self, ctx: RunContext) -> dict[str, list]:
        return {"version": ctx.feature_versions()}

    def validate(self, p: ExtractParams, ctx: RunContext) -> list[str]:
        errors = check_version(p.version, ctx)
        candidates = TOOLS if p.tools is None else p.tools
        col_tools = [t for t in candidates if _pipeline_flags(t)[0]]
        opt_tools = [t for t in candidates if _pipeline_flags(t)[1]]
        errors += [
            f"feature_col: {k!r} is not a tool of this step that uses a column "
            f"({', '.join(col_tools) or 'none'})"
            for k in p.feature_col
            if k not in col_tools
        ]
        for name, value in (("binary", p.binary), ("min_patients", p.min_patients)):
            if isinstance(value, dict):
                errors += [
                    f"{name}: {k!r} is not a tool of this step that uses matrix options "
                    f"({', '.join(opt_tools) or 'none'})"
                    for k in value
                    if k not in opt_tools
                ]
        mins = p.min_patients.values() if isinstance(p.min_patients, dict) else [p.min_patients]
        errors += [
            f"min_patients must be 1-{MAX_MIN_PATIENTS}, got {m}"
            for m in mins
            if m is not None and not 1 <= m <= MAX_MIN_PATIENTS
        ]
        return errors

    def check_inputs(self, p: ExtractParams, ctx: RunContext) -> list[str]:
        import pandas as pd

        root = ctx.stage_dir(p.version, "preprocessed")
        tools = _present_tools(ctx, p.version, "preprocessed", p.tools)
        if not tools:
            return [f"no preprocessed data under {root}"]
        errors = []
        for tool in (t for t in tools if _pipeline_flags(t)[0]):
            col = p.feature_col.get(tool)
            if col is None:
                errors.append(f"feature_col: give a column for {tool}")
                continue
            for f in sorted((root / tool).glob("*.csv")):
                columns = list(pd.read_csv(f, sep=";", nrows=0).columns)
                if col not in columns:
                    errors.append(
                        f"feature_col: {col!r} is not a column of {tool}/{f.name} "
                        f"(columns: {', '.join(c for c in columns if c != 'Accession')})"
                    )
        return errors

    def label(self, p: ExtractParams) -> str:
        return p.version

    def inputs(self, p: ExtractParams, ctx: RunContext) -> list[Path]:
        return _tool_dirs(ctx, p.version, "preprocessed", p.tools)

    def writes(self, p: ExtractParams, ctx: RunContext) -> list[Path]:
        return _tool_dirs(ctx, p.version, "extracted", p.tools)

    def run(self, p: ExtractParams, ctx: RunContext) -> list[Path]:
        from phantom.features.extraction.extractor import FeatureExtractor

        return FeatureExtractor(
            p.version,
            ctx.config_mgr,
            feature_col=p.feature_col,
            binary=p.binary,
            min_patients=p.min_patients,
            tools=_present_tools(ctx, p.version, "preprocessed", p.tools),
        ).run()
