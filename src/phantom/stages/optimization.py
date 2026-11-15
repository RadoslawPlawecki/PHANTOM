"""
``features.optimize`` -- Optuna search over feature-engineering choices,
evaluated exploratory (optimistic), by patient-level nested CV (deleaked) and
by a max-statistic permutation test (significance). Replaces Features ->
Optimize features.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from phantom.config.loader import ConfigLoader
from phantom.runner.parameters import param
from phantom.runner.stage import RunContext, Stage, register
from phantom.stages._common import (
    MODEL_HELP,
    SEED_HELP,
    VIRUS_IDENTIFICATION_TOOL_HELP,
    Model,
    check_metadata,
    check_version,
    check_virus_identification_tool,
    metadata_path,
    output_dir,
    resolve_virus_identification_tool,
)

OPTIMIZATION_DIR = "optimization"
NESTED_KNOBS = ("outer_folds", "outer_repeats", "inner_trials", "top_k_features")
PERMUTATION_KNOBS = ("n_permutations", "permutation_trials")


@dataclass(frozen=True)
class OptimizeParams:
    version: str = param(help="feature version whose raw-merged data is searched")
    tool: Literal["cherry", "phagcn", "phavip"] = param(grid=True)
    vtool: str = param(
        help=VIRUS_IDENTIFICATION_TOOL_HELP + "; whose merged file to search", grid=True
    )
    mode: Literal["exploratory", "nested", "permutation", "all"] = param(
        "exploratory",
        help="exploratory = optimistic CV on all data; nested = deleaked estimate; "
        "permutation = significance; all = the three in turn",
    )
    mask: str | None = param(
        None, help='CheckV mask in <[masks]>/<vtool>/, e.g. "MQ_vMAGs.tsv"; default: none'
    )
    model: Model = param("catboost", help=MODEL_HELP, grid=True)
    validator: Literal["loocv", "rcv"] = param("loocv")
    metric: Literal["mcc", "gmean"] = param("mcc")
    smote: bool = param(False)
    seed: int = param(42, help=SEED_HELP, grid=True)
    outer_folds: int = param(5, help="nested: outer folds")
    outer_repeats: int = param(3, help="nested: outer repeats")
    inner_trials: int | None = param(
        None, help="nested: trials per outer fold; default: tool budget"
    )
    top_k_features: int = param(10, help="nested: features in the ablation model")
    n_permutations: int = param(50, help="permutation: label shuffles")
    permutation_trials: int | None = param(
        None, help="permutation: trials per shuffle; default: tool budget"
    )
    out_dir: str | None = param(
        None,
        help="default: <[results]>/optimization/<version>/<tool>/<vtool>; "
        "relative = under the workspace",
    )


@register
class Optimize(Stage):
    name = "features.optimize"
    summary = (
        "Optuna search over feature-engineering choices (exploratory / nested CV / permutation)"
    )
    Params = OptimizeParams

    def resolve(self, params, context: RunContext):
        return dataclasses.replace(
            params, vtool=resolve_virus_identification_tool(context, params.vtool)
        )

    def defaults(self, context: RunContext) -> dict:
        return {"vtool": context.virus_identification_tools}

    def choices(self, context: RunContext) -> dict[str, list]:
        return {"version": context.feature_versions(), "vtool": context.virus_identification_tools}

    def validate(self, params: OptimizeParams, context: RunContext) -> list[str]:
        errors = check_version(params.version, context) + check_virus_identification_tool(
            params.vtool, context
        )
        for name in (*NESTED_KNOBS, *PERMUTATION_KNOBS):
            value = getattr(params, name)
            if value is not None and value < 1:
                errors.append(f"{name} must be >= 1, got {value}")
        from phantom.analysis.classification.machine_learning.models import NO_BUILTIN_IMPORTANCE

        if params.model in NO_BUILTIN_IMPORTANCE and params.mode in ("nested", "all"):
            errors.append(
                f"model {params.model!r} has no built-in feature importances, which the nested-CV "
                "top-K ablation needs: use mode = exploratory or permutation, or another model"
            )
        if params.outer_folds < 2:
            errors.append("outer_folds must be >= 2")
        if params.mask and not context.config.get("masks", {}).get("path"):
            errors.append("config.toml is missing [masks].path")
        return errors

    def _merged_file(self, params: OptimizeParams, context: RunContext) -> Path | None:
        folder = context.stage_dir(params.version, "raw_merged") / params.tool
        return next(iter(sorted(folder.glob(f"{params.vtool}_*.csv"))), None)

    def _mask(self, params: OptimizeParams, context: RunContext) -> Path | None:
        if not params.mask:
            return None
        tool_name = ConfigLoader.virus_identification_tools(context.config)[params.vtool]
        return context.section_path("masks") / tool_name / params.mask

    def check_inputs(self, params: OptimizeParams, context: RunContext) -> list[str]:
        errors = []
        if self._merged_file(params, context) is None:
            folder = context.stage_dir(params.version, "raw_merged") / params.tool
            errors.append(f"no {params.vtool}_*.csv merged file in {folder}")
        mask = self._mask(params, context)
        if mask is not None and not mask.is_file():
            errors.append(f"mask not found: {mask}")
        return errors + check_metadata(context)

    def label(self, params: OptimizeParams) -> str:
        return f"{params.tool} / {params.vtool} / {params.mode} / {params.model}"

    def inputs(self, params: OptimizeParams, context: RunContext) -> list[Path]:
        merged = (
            self._merged_file(params, context)
            or context.stage_dir(params.version, "raw_merged") / params.tool
        )
        mask = self._mask(params, context)
        return [merged, *([mask] if mask else []), metadata_path(context)]

    def _out_dir(self, params: OptimizeParams, context: RunContext) -> Path:
        return output_dir(
            params.out_dir, context, OPTIMIZATION_DIR, params.version, params.tool, params.vtool
        )

    def writes(self, params: OptimizeParams, context: RunContext) -> list[Path]:
        return [self._out_dir(params, context)]

    def run(self, params: OptimizeParams, context: RunContext) -> list[Path]:
        from phantom.analysis.features.optimization.optimizer import run_optimization
        from phantom.analysis.features.optimization.search import RunConfig, load_masked

        source = self._merged_file(params, context)
        if source is None:  # checked before running; kept for type narrowing
            raise FileNotFoundError(f"no merged {params.tool} file for {params.vtool}")
        # Knobs of strategies the mode does not run are recorded as None.
        nested = params.mode in ("nested", "all")
        permutation = params.mode in ("permutation", "all")
        knobs = {
            **{k: getattr(params, k) if nested else None for k in NESTED_KNOBS},
            **{k: getattr(params, k) if permutation else None for k in PERMUTATION_KNOBS},
        }
        out_dir = self._out_dir(params, context)
        run_optimization(
            params.tool,
            params.mode,
            RunConfig(
                model=params.model,
                validator=params.validator,
                metric=params.metric,
                smote=params.smote,
                seed=params.seed,
            ),
            out_dir,
            source,
            load_masked(params.tool, source, self._mask(params, context)),
            **knobs,
        )
        return [out_dir]
