"""Helpers shared by stage adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from phantom.analysis.genomes.metadata.table import MetadataTable
from phantom.config.loader import ConfigLoader
from phantom.runner.stage import RunContext

# The algorithms of phantom.analysis.classification.machine_learning.models.MODELS (a test keeps them equal).
Model = Literal["catboost", "rf", "xgb", "logreg", "hgb"]
MODEL_HELP = (
    "catboost | rf (random forest) | xgb (XGBoost) | logreg (logistic regression) | "
    "hgb (histogram gradient boosting); a list = one job per model"
)

SEED_HELP = (
    "random seed of everything random in the job (models, SMOTE, CV splits, permutation "
    "importance, Optuna, MDS, PERMANOVA); a list = one job per seed"
)


def check_version(version: str, context: RunContext) -> list[str]:
    """Declared in config.toml, or created by an earlier step of this run."""
    if context.version_known(version):
        return []
    versions = context.feature_versions()
    return [f"version {version!r} not in config.toml (available: {', '.join(versions)})"]


VIRUS_IDENTIFICATION_TOOL_HELP = (
    "a virus identification tool: its abbreviation in [virus-identification-tools] "
    '(e.g. "geN") or its full name (e.g. "genomad")'
)


def resolve_virus_identification_tool(context: RunContext, virus_identification_tool: str) -> str:
    """A tool's full name -> its abbreviation; anything unknown is left for check_vtool."""
    return (
        ConfigLoader.virus_identification_tool_abbreviation(
            context.config, virus_identification_tool
        )
        or virus_identification_tool
    )


def check_virus_identification_tool(
    virus_identification_tool: str, context: RunContext
) -> list[str]:
    if virus_identification_tool in context.virus_identification_tools:
        return []
    names = ", ".join(
        f"{abbreviation} ({name})"
        for abbreviation, name in ConfigLoader.virus_identification_tools(context.config).items()
    )
    return [
        f"vtool {virus_identification_tool!r} is not a virus identification tool (available: {names})"
    ]


def output_dir(out_dir: str | None, context: RunContext, *default_parts: str) -> Path:
    """An explicit ``out_dir`` parameter, else ``<[results]>/<default_parts...>``."""
    if out_dir:
        return context.resolve(out_dir)
    return context.section_path("results").joinpath(*default_parts)


def metadata_path(context: RunContext) -> Path:
    return MetadataTable.path_from_config(context.config)


def check_metadata(context: RunContext) -> list[str]:
    """The class labels a classification needs: the metadata CSV exists, has a
    label for its samples (0/1, ids unique) and both classes are present."""
    from phantom.analysis.classification.data.labeling import Labeling

    path = metadata_path(context)
    if not path.is_file():
        return [f"metadata with class labels not found: {path}"]
    try:
        labels = Labeling.from_metadata(path).labels
    except (ValueError, KeyError) as e:
        return [f"{path}: {e.args[0] if e.args else e}"]
    if labels.nunique() < 2:
        counts = labels.value_counts().to_dict()
        return [f"{path}: classification needs cases (1) and controls (0), got {counts or 'none'}"]
    return []
