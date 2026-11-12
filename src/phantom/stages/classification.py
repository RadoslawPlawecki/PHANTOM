"""
Classification stages.

``classification.single``  one supervised classifier per (modality, vtool, model)
``classification.multi``   early/late fusion of the comp/host/func modalities
``classification.mds``     unsupervised MDS exploration of the three modalities
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from phantom.analysis.classification.data.inputs import find_feature_file
from phantom.runner.params import param
from phantom.runner.stage import RunContext, Stage, register
from phantom.stages._common import (
    check_metadata,
    check_version,
    check_vtool,
    metadata_path,
    output_dir,
)

# Subdirectory of [results] per run type.
SINGLE_MODAL_DIR = "single-modal"
MULTI_MODAL_DIR = "multi-modal"
MDS_DIR = "mds"
# The fusion/MDS executors read exactly these three modality keys.
FUSED_MODALITIES = ("comp", "host", "func")

Model = Literal["catboost", "rf", "xgb"]
Validator = Literal["loocv", "rcv"]
Fusion = Literal["early", "late"]

VERSION_HELP = "feature version, a [features.vX_Y] block in config.toml"
VTOOL_HELP = "a key of [tools] in config.toml"
SMOTE_HELP = "SMOTE oversampling inside each training fold"


def _check_validators(validators: Sequence[str]) -> list[str]:
    if not validators:
        return ["validators: give at least one of loocv, rcv"]
    if len(set(validators)) != len(validators):
        return [f"validators: duplicates in {validators}"]
    return []


def _feature_file(ctx: RunContext, version: str, modality: str, vtool: str) -> Path:
    return find_feature_file(ctx.config_mgr, version, ctx.modalities[modality], vtool)


def _feature_input(ctx: RunContext, version: str, modality: str, vtool: str) -> Path:
    """The feature file if it exists, else the directory an extraction job
    will write it into (so a classification after an extraction in the same
    run file is planned as depending on it)."""
    try:
        return _feature_file(ctx, version, modality, vtool)
    except (FileNotFoundError, ValueError):
        cfg = ctx.modalities[modality]
        return ctx.stage_dir(version, "extracted") / cfg["tool"]


def _check_files(ctx: RunContext, version: str, modalities, vtool: str) -> list[str]:
    errors = []
    for modality in modalities:
        try:
            _feature_file(ctx, version, modality, vtool)
        except FileNotFoundError as e:
            errors.append(str(e))
    return errors + check_metadata(ctx)


def _check_fused_modalities(ctx: RunContext) -> list[str]:
    missing = [m for m in FUSED_MODALITIES if m not in ctx.modalities]
    if missing:
        return [
            f"[modalities] must define {', '.join(FUSED_MODALITIES)} (missing: {', '.join(missing)})"
        ]
    return []


# --- classification.single ---------------------------------------------------


@dataclass(frozen=True)
class SingleModalParams:
    version: str = param(help=VERSION_HELP)
    modality: str = param(help="a key of [modalities] in config.toml", grid=True)
    vtool: str = param(help=VTOOL_HELP, grid=True)
    model: Model = param("catboost", grid=True)
    validators: list[Validator] = param(default_factory=lambda: ["loocv"])
    smote: bool = param(False, help=SMOTE_HELP)
    out_dir: str | None = param(
        None, help="default: <[results]>/single-modal/<version>; relative = under the workspace"
    )


@register
class SingleModal(Stage):
    name = "classification.single"
    summary = "Single-modal classifier per modality x virus identification tool x model"
    Params = SingleModalParams

    def defaults(self, ctx: RunContext) -> dict:
        return {"vtool": ctx.vtools}

    def choices(self, ctx: RunContext) -> dict[str, list]:
        return {
            "version": ctx.feature_versions(),
            "modality": list(ctx.modalities),
            "vtool": ctx.vtools,
        }

    def validate(self, p: SingleModalParams, ctx: RunContext) -> list[str]:
        errors = check_version(p.version, ctx) + check_vtool(p.vtool, ctx)
        if p.modality not in ctx.modalities:
            errors.append(
                f"modality {p.modality!r} not in [modalities] "
                f"(available: {', '.join(ctx.modalities)})"
            )
        return errors + _check_validators(p.validators)

    def check_inputs(self, p: SingleModalParams, ctx: RunContext) -> list[str]:
        return _check_files(ctx, p.version, [p.modality], p.vtool)

    def label(self, p: SingleModalParams) -> str:
        return f"{p.modality} / {p.vtool} / {p.model}"

    def inputs(self, p: SingleModalParams, ctx: RunContext) -> list[Path]:
        # The labels are an input too: relabeling must invalidate results.
        return [_feature_input(ctx, p.version, p.modality, p.vtool), metadata_path(ctx)]

    def writes(self, p: SingleModalParams, ctx: RunContext) -> list[Path]:
        return [output_dir(p.out_dir, ctx, SINGLE_MODAL_DIR, p.version)]

    def run(self, p: SingleModalParams, ctx: RunContext) -> list[Path]:
        from phantom.analysis.classification.execution.config import SingleModalConfig
        from phantom.analysis.classification.execution.single_modal import (
            SingleModalClassifier,
        )

        out_dir = output_dir(p.out_dir, ctx, SINGLE_MODAL_DIR, p.version)
        out_dir.mkdir(parents=True, exist_ok=True)
        classifier = SingleModalClassifier(
            SingleModalConfig(
                in_file=str(_feature_file(ctx, p.version, p.modality, p.vtool)),
                out_dir=str(out_dir),
                modality=p.modality,
                vtool=p.vtool,
                run_loocv="loocv" in p.validators,
                run_repeated="rcv" in p.validators,
                use_smote=p.smote,
                model_type=p.model,
            )
        )
        classifier.run()
        return [Path(classifier.saver.exp_dir)]


# --- classification.multi ----------------------------------------------------


@dataclass(frozen=True)
class MultiModalParams:
    version: str = param(help=VERSION_HELP)
    vtool: str = param(help=VTOOL_HELP, grid=True)
    fusion: Fusion = param("early", grid=True)
    model: Model = param("catboost", grid=True)
    validators: list[Validator] = param(default_factory=lambda: ["loocv"])
    smote: bool = param(False, help=SMOTE_HELP)
    optimize_weights: bool = param(
        False, help="late fusion only: tune per-modality weights with Optuna"
    )
    n_trials: int = param(30, help="Optuna trials when optimize_weights = true")
    out_dir: str | None = param(
        None, help="default: <[results]>/multi-modal/<version>; relative = under the workspace"
    )


@register
class MultiModal(Stage):
    name = "classification.multi"
    summary = "Multimodal (comp + host + func) early/late fusion classifier per vtool"
    Params = MultiModalParams

    def defaults(self, ctx: RunContext) -> dict:
        return {"vtool": ctx.vtools}

    def choices(self, ctx: RunContext) -> dict[str, list]:
        return {"version": ctx.feature_versions(), "vtool": ctx.vtools}

    def validate(self, p: MultiModalParams, ctx: RunContext) -> list[str]:
        errors = check_version(p.version, ctx) + check_vtool(p.vtool, ctx)
        errors += _check_validators(p.validators)
        if p.optimize_weights and p.fusion != "late":
            errors.append("optimize_weights applies to fusion = 'late' only")
        if p.n_trials < 1:
            errors.append(f"n_trials must be >= 1, got {p.n_trials}")
        return errors + _check_fused_modalities(ctx)

    def check_inputs(self, p: MultiModalParams, ctx: RunContext) -> list[str]:
        return _check_files(ctx, p.version, FUSED_MODALITIES, p.vtool)

    def label(self, p: MultiModalParams) -> str:
        weights = " (tuned weights)" if p.optimize_weights else ""
        return f"{p.vtool} / {p.fusion}{weights} / {p.model}"

    def inputs(self, p: MultiModalParams, ctx: RunContext) -> list[Path]:
        files = [_feature_input(ctx, p.version, m, p.vtool) for m in FUSED_MODALITIES]
        return [*files, metadata_path(ctx)]

    def writes(self, p: MultiModalParams, ctx: RunContext) -> list[Path]:
        return [output_dir(p.out_dir, ctx, MULTI_MODAL_DIR, p.version)]

    def run(self, p: MultiModalParams, ctx: RunContext) -> list[Path]:
        from phantom.analysis.classification.execution.config import MultiModalConfig
        from phantom.analysis.classification.execution.multi_modal import MultiModalClassifier

        out_dir = output_dir(p.out_dir, ctx, MULTI_MODAL_DIR, p.version)
        out_dir.mkdir(parents=True, exist_ok=True)
        comp, host, func = (
            str(_feature_file(ctx, p.version, m, p.vtool)) for m in FUSED_MODALITIES
        )
        # No weight tuning is (False, 0) for the executor.
        tune = p.fusion == "late" and p.optimize_weights
        classifier = MultiModalClassifier(
            MultiModalConfig(
                comp=comp,
                host=host,
                func=func,
                out_dir=str(out_dir),
                run_loocv="loocv" in p.validators,
                run_repeated="rcv" in p.validators,
                use_smote=p.smote,
                model_type=p.model,
                fusion=p.fusion,
                opt=tune,  # type: ignore[arg-type]  # the executor treats it as a flag
                n_trials=p.n_trials if tune else 0,
            )
        )
        classifier.run()
        return [Path(classifier.saver.exp_dir)]


# --- classification.mds ------------------------------------------------------


@dataclass(frozen=True)
class MdsParams:
    version: str = param(help=VERSION_HELP)
    vtool: str = param(help=VTOOL_HELP, grid=True)
    out_dir: str | None = param(
        None, help="default: <[results]>/mds/<version>; relative = under the workspace"
    )


@register
class Mds(Stage):
    name = "classification.mds"
    summary = "Unsupervised MDS exploration of comp/host/func per vtool"
    Params = MdsParams

    def defaults(self, ctx: RunContext) -> dict:
        return {"vtool": ctx.vtools}

    def choices(self, ctx: RunContext) -> dict[str, list]:
        return {"version": ctx.feature_versions(), "vtool": ctx.vtools}

    def validate(self, p: MdsParams, ctx: RunContext) -> list[str]:
        return (
            check_version(p.version, ctx) + check_vtool(p.vtool, ctx) + _check_fused_modalities(ctx)
        )

    def check_inputs(self, p: MdsParams, ctx: RunContext) -> list[str]:
        return _check_files(ctx, p.version, FUSED_MODALITIES, p.vtool)

    def label(self, p: MdsParams) -> str:
        return p.vtool

    def inputs(self, p: MdsParams, ctx: RunContext) -> list[Path]:
        files = [_feature_input(ctx, p.version, m, p.vtool) for m in FUSED_MODALITIES]
        return [*files, metadata_path(ctx)]

    def writes(self, p: MdsParams, ctx: RunContext) -> list[Path]:
        return [output_dir(p.out_dir, ctx, MDS_DIR, p.version)]

    def run(self, p: MdsParams, ctx: RunContext) -> list[Path]:
        from phantom.analysis.classification.execution.config import MdsConfig
        from phantom.analysis.classification.execution.mds import (
            MdsAnalysis,
        )

        out_dir = output_dir(p.out_dir, ctx, MDS_DIR, p.version)
        out_dir.mkdir(parents=True, exist_ok=True)
        comp, host, func = (
            str(_feature_file(ctx, p.version, m, p.vtool)) for m in FUSED_MODALITIES
        )
        classifier = MdsAnalysis(MdsConfig(comp=comp, host=host, func=func, out_dir=str(out_dir)))
        classifier.run()
        return [Path(classifier.saver.exp_dir)]
