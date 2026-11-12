"""
``features.optimize`` -- Optuna search over feature-engineering choices,
evaluated exploratory (optimistic), by patient-level nested CV (deleaked) and
by a max-statistic permutation test (significance). Replaces Features ->
Optimize features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from phantom.runner.params import param
from phantom.runner.stage import RunContext, Stage, register
from phantom.stages._common import (
    check_metadata,
    check_version,
    check_vtool,
    metadata_path,
    output_dir,
)

OPTIMIZATION_DIR = "optimization"
NESTED_KNOBS = ("outer_folds", "outer_repeats", "inner_trials", "top_k_features")
PERMUTATION_KNOBS = ("n_permutations", "permutation_trials")


@dataclass(frozen=True)
class OptimizeParams:
    version: str = param(help="feature version whose raw-merged data is searched")
    tool: Literal["cherry", "phagcn", "phavip"] = param(grid=True)
    vtool: str = param(help="a key of [tools]: whose merged file to search", grid=True)
    mode: Literal["exploratory", "nested", "permutation", "all"] = param(
        "exploratory",
        help="exploratory = optimistic CV on all data; nested = deleaked estimate; "
        "permutation = significance; all = the three in turn",
    )
    mask: str | None = param(
        None, help='CheckV mask in <[masks]>/<vtool>/, e.g. "MQ_vMAGs.tsv"; default: none'
    )
    model: Literal["catboost", "rf", "xgb"] = param("catboost", grid=True)
    validator: Literal["loocv", "rcv"] = param("loocv")
    metric: Literal["mcc", "gmean"] = param("mcc")
    smote: bool = param(False)
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

    def defaults(self, ctx: RunContext) -> dict:
        return {"vtool": ctx.vtools}

    def choices(self, ctx: RunContext) -> dict[str, list]:
        return {"version": ctx.feature_versions(), "vtool": ctx.vtools}

    def validate(self, p: OptimizeParams, ctx: RunContext) -> list[str]:
        errors = check_version(p.version, ctx) + check_vtool(p.vtool, ctx)
        for name in (*NESTED_KNOBS, *PERMUTATION_KNOBS):
            value = getattr(p, name)
            if value is not None and value < 1:
                errors.append(f"{name} must be >= 1, got {value}")
        if p.outer_folds < 2:
            errors.append("outer_folds must be >= 2")
        if p.mask and not ctx.config.get("masks", {}).get("path"):
            errors.append("config.toml is missing [masks].path")
        return errors

    def _merged_file(self, p: OptimizeParams, ctx: RunContext) -> Path | None:
        folder = ctx.stage_dir(p.version, "raw_merged") / p.tool
        return next(iter(sorted(folder.glob(f"{p.vtool}_*.csv"))), None)

    def _mask(self, p: OptimizeParams, ctx: RunContext) -> Path | None:
        if not p.mask:
            return None
        tool_name = ctx.config["tools"][p.vtool]
        return ctx.section_path("masks") / tool_name / p.mask

    def check_inputs(self, p: OptimizeParams, ctx: RunContext) -> list[str]:
        errors = []
        if self._merged_file(p, ctx) is None:
            folder = ctx.stage_dir(p.version, "raw_merged") / p.tool
            errors.append(f"no {p.vtool}_*.csv merged file in {folder}")
        mask = self._mask(p, ctx)
        if mask is not None and not mask.is_file():
            errors.append(f"mask not found: {mask}")
        return errors + check_metadata(ctx)

    def label(self, p: OptimizeParams) -> str:
        return f"{p.tool} / {p.vtool} / {p.mode} / {p.model}"

    def inputs(self, p: OptimizeParams, ctx: RunContext) -> list[Path]:
        merged = self._merged_file(p, ctx) or ctx.stage_dir(p.version, "raw_merged") / p.tool
        mask = self._mask(p, ctx)
        return [merged, *([mask] if mask else []), metadata_path(ctx)]

    def _out_dir(self, p: OptimizeParams, ctx: RunContext) -> Path:
        return output_dir(p.out_dir, ctx, OPTIMIZATION_DIR, p.version, p.tool, p.vtool)

    def writes(self, p: OptimizeParams, ctx: RunContext) -> list[Path]:
        return [self._out_dir(p, ctx)]

    def run(self, p: OptimizeParams, ctx: RunContext) -> list[Path]:
        from phantom.analysis.features.optimization.optimizer import run_optimization
        from phantom.analysis.features.optimization.search import RunConfig, load_masked

        source = self._merged_file(p, ctx)
        if source is None:  # checked before running; kept for type narrowing
            raise FileNotFoundError(f"no merged {p.tool} file for {p.vtool}")
        # Knobs of strategies the mode does not run are recorded as None.
        nested = p.mode in ("nested", "all")
        permutation = p.mode in ("permutation", "all")
        knobs = {
            **{k: getattr(p, k) if nested else None for k in NESTED_KNOBS},
            **{k: getattr(p, k) if permutation else None for k in PERMUTATION_KNOBS},
        }
        out_dir = self._out_dir(p, ctx)
        run_optimization(
            p.tool,
            p.mode,
            RunConfig(model=p.model, validator=p.validator, metric=p.metric, smote=p.smote),
            out_dir,
            source,
            load_masked(p.tool, source, self._mask(p, ctx)),
            **knobs,
        )
        return [out_dir]
