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

from phantom.config.loader import ConfigLoader
from phantom.runner.parameters import param
from phantom.runner.stage import RunContext, Stage, register
from phantom.stages._common import check_version

VERSION_HELP = "feature version, a [features.vX_Y] block in config.toml"
Tool = Literal["cherry", "phagcn", "phavip", "phatyp"]
TOOLS: tuple[str, ...] = ("cherry", "phagcn", "phavip", "phatyp")
MAX_MIN_PATIENTS = 10  # upper bound for min_patients


def _pipeline_flags(tool: str) -> tuple[bool, bool]:
    """(needs a feature column, uses binary/min_patients) for a tool."""
    from phantom.analysis.features.extractor import PIPELINES

    cls = PIPELINES[tool]
    return bool(getattr(cls, "NEEDS_FEATURE_COLUMN", False)), bool(
        getattr(cls, "NEEDS_MATRIX_OPTIONS", False)
    )


def _tool_dirs(
    context: RunContext, version: str, stage: str, tools: Sequence[str] | None
) -> list[Path]:
    """The stage dir itself (all tools) or one subdirectory per requested tool."""
    root = context.stage_dir(version, stage)
    return [root] if tools is None else [root / t for t in tools]


def _present_tools(
    context: RunContext, version: str, stage: str, tools: Sequence[str] | None
) -> list[str]:
    root = context.stage_dir(version, stage)
    wanted = TOOLS if tools is None else tools
    return [t for t in TOOLS if t in wanted and (root / t).is_dir()]


# features.import


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

    def announces(self, params: ImportParams) -> list[str]:
        return [params.version] if params.create else []

    def validate(self, params: ImportParams, context: RunContext) -> list[str]:
        errors = []
        # Rerunning the step that created the version is fine; reusing a
        # version some other import created is refused.
        if (
            params.create
            and params.version in context.feature_versions()
            and not context.ran_before(self.name, params)
        ):
            errors.append(
                f"version {params.version!r} already exists; use create = false to add to it, "
                "or choose a new version number"
            )
        if not params.create:
            errors += check_version(params.version, context)
        if not params.sources:
            errors.append("sources: give at least one path")
        return errors

    def check_inputs(self, params: ImportParams, context: RunContext) -> list[str]:
        return [f"source not found: {s}" for s in params.sources if not context.resolve(s).exists()]

    def label(self, params: ImportParams) -> str:
        return f"{len(params.sources)} source(s) -> {params.version}"

    def inputs(self, params: ImportParams, context: RunContext) -> list[Path]:
        return [context.resolve(s) for s in params.sources]

    def writes(self, params: ImportParams, context: RunContext) -> list[Path]:
        return [context.stage_dir(params.version, "raw")]

    def run(self, params: ImportParams, context: RunContext) -> list[Path]:
        from phantom.analysis.features.importer import FeatureImporter

        importer = FeatureImporter(config_path=context.config_path)
        create = params.create and params.version not in context.feature_versions()  # else: a rerun
        raw_dir, failed = importer.import_sources(
            [context.resolve(s) for s in params.sources],
            params.version,
            create=create,
            overwrite=params.overwrite,
        )
        if failed:
            raise RuntimeError("could not copy: " + "; ".join(failed))
        if create:
            context.reload()  # config.toml gained a [features.vX_Y] block
        return [raw_dir]


# features.merge


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

    def choices(self, context: RunContext) -> dict[str, list]:
        return {"version": context.feature_versions()}

    def _merger(self, params: MergeParams, context: RunContext):
        from phantom.analysis.features.merger import FeatureMerger

        return FeatureMerger(params.version, context.feature_version_manager, params.id_columns)

    def validate(self, params: MergeParams, context: RunContext) -> list[str]:
        errors = check_version(params.version, context)
        if not params.id_columns:
            errors.append('id_columns: give at least one pattern, e.g. { "*" = "Accession" }')
        return errors

    def check_inputs(self, params: MergeParams, context: RunContext) -> list[str]:
        merger = self._merger(params, context)
        leaves = merger.leaf_dirs()
        if not leaves:
            return [f"no .tsv/.csv files under {merger.raw_dir}"]
        errors = []
        for leaf in leaves:
            label = str(leaf.relative_to(merger.raw_dir))
            first = merger.raw_files(leaf)[0]
            columns = merger.peek_columns(first)
            column = merger.id_column_for(label)
            if column is None:
                errors.append(f"id_columns: no pattern matches raw/{label}")
            elif column not in columns:
                errors.append(
                    f"id_columns: {column!r} (for raw/{label}) is not a column of {first.name} "
                    f"(columns: {', '.join(columns)})"
                )
        return errors

    def label(self, params: MergeParams) -> str:
        return params.version

    def inputs(self, params: MergeParams, context: RunContext) -> list[Path]:
        return [context.stage_dir(params.version, "raw")]

    def writes(self, params: MergeParams, context: RunContext) -> list[Path]:
        return [context.stage_dir(params.version, "raw_merged")]

    def run(self, params: MergeParams, context: RunContext) -> list[Path]:
        written = self._merger(params, context).run()
        return sorted({f.parent for f in written})


# features.preprocess


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

    def choices(self, context: RunContext) -> dict[str, list]:
        masks = context.section_path("masks")
        names = (
            sorted({mask_file.name for mask_file in masks.glob("*/*.tsv")})
            if masks.is_dir()
            else []
        )
        return {"version": context.feature_versions(), "mask": names}

    def _mask_files(self, params: PreprocessParams, context: RunContext) -> list[Path]:
        """The mask file of every configured vtool (one per vtool directory)."""
        if not params.mask:
            return []
        root = context.section_path("masks")
        return [
            root / tool_name / params.mask
            for tool_name in ConfigLoader.virus_identification_tools(context.config).values()
        ]

    def validate(self, params: PreprocessParams, context: RunContext) -> list[str]:
        errors = check_version(params.version, context)
        if params.mask and not context.config.get("masks", {}).get("path"):
            errors.append("config.toml is missing [masks].path")
        return errors

    def check_inputs(self, params: PreprocessParams, context: RunContext) -> list[str]:
        from phantom.analysis.features.preprocessor import mask_dir_for

        root = context.stage_dir(params.version, "raw_merged")
        files = [
            f
            for t in _present_tools(context, params.version, "raw_merged", params.tools)
            for f in sorted((root / t).glob("*.csv"))
        ]
        if not files:
            wanted = ", ".join(params.tools) if params.tools else "any tool"
            return [f"no merged files for {wanted} under {root}"]
        errors = []
        for f in files:
            try:
                mask_dir = mask_dir_for(f, context.config)
            except ValueError as e:
                errors.append(str(e))
                continue
            if params.mask and not (mask_dir / params.mask).is_file():
                errors.append(f"mask not found: {mask_dir / params.mask}")
        return errors

    def label(self, params: PreprocessParams) -> str:
        return f"{params.version} / mask: {params.mask or 'none'}"

    def inputs(self, params: PreprocessParams, context: RunContext) -> list[Path]:
        return [
            *_tool_dirs(context, params.version, "raw_merged", params.tools),
            *self._mask_files(params, context),
        ]

    def writes(self, params: PreprocessParams, context: RunContext) -> list[Path]:
        return _tool_dirs(context, params.version, "preprocessed", params.tools)

    def run(self, params: PreprocessParams, context: RunContext) -> list[Path]:
        from phantom.analysis.features.preprocessor import FeaturePreprocessor

        return FeaturePreprocessor(
            params.version,
            context.config,
            context.feature_version_manager,
            mask=params.mask,
            tools=_present_tools(context, params.version, "raw_merged", params.tools),
        ).run()


# features.extract


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

    def choices(self, context: RunContext) -> dict[str, list]:
        return {"version": context.feature_versions()}

    def validate(self, params: ExtractParams, context: RunContext) -> list[str]:
        errors = check_version(params.version, context)
        candidates = TOOLS if params.tools is None else params.tools
        col_tools = [t for t in candidates if _pipeline_flags(t)[0]]
        opt_tools = [t for t in candidates if _pipeline_flags(t)[1]]
        errors += [
            f"feature_col: {k!r} is not a tool of this step that uses a column "
            f"({', '.join(col_tools) or 'none'})"
            for k in params.feature_col
            if k not in col_tools
        ]
        for name, value in (("binary", params.binary), ("min_patients", params.min_patients)):
            if isinstance(value, dict):
                errors += [
                    f"{name}: {k!r} is not a tool of this step that uses matrix options "
                    f"({', '.join(opt_tools) or 'none'})"
                    for k in value
                    if k not in opt_tools
                ]
        mins = (
            params.min_patients.values()
            if isinstance(params.min_patients, dict)
            else [params.min_patients]
        )
        errors += [
            f"min_patients must be 1-{MAX_MIN_PATIENTS}, got {minimum}"
            for minimum in mins
            if minimum is not None and not 1 <= minimum <= MAX_MIN_PATIENTS
        ]
        return errors

    def check_inputs(self, params: ExtractParams, context: RunContext) -> list[str]:
        import pandas as pd

        root = context.stage_dir(params.version, "preprocessed")
        tools = _present_tools(context, params.version, "preprocessed", params.tools)
        if not tools:
            return [f"no preprocessed data under {root}"]
        errors = []
        for tool in (t for t in tools if _pipeline_flags(t)[0]):
            col = params.feature_col.get(tool)
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

    def label(self, params: ExtractParams) -> str:
        return params.version

    def inputs(self, params: ExtractParams, context: RunContext) -> list[Path]:
        return _tool_dirs(context, params.version, "preprocessed", params.tools)

    def writes(self, params: ExtractParams, context: RunContext) -> list[Path]:
        return _tool_dirs(context, params.version, "extracted", params.tools)

    def run(self, params: ExtractParams, context: RunContext) -> list[Path]:
        from phantom.analysis.features.extractor import FeatureExtractor

        return FeatureExtractor(
            params.version,
            context.feature_version_manager,
            feature_col=params.feature_col,
            binary=params.binary,
            min_patients=params.min_patients,
            tools=_present_tools(context, params.version, "preprocessed", params.tools),
        ).run()
