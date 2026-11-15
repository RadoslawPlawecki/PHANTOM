"""
Classification stages.

``classification.single``  one supervised classifier per (modality, vtool, model)
``classification.multi``   early/late fusion of the comp/host/func modalities
``classification.mds``     unsupervised MDS exploration of the three modalities
``classification.figures`` an experiment's figures and metrics table (newest run per setting)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from phantom.analysis.classification.data.loading import find_feature_file
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

# Subdirectory of [results] per run type.
SINGLE_MODAL_DIR = "single-modal"
MULTI_MODAL_DIR = "multi-modal"
MDS_DIR = "mds"
# The fusion/MDS executors read exactly these three modality keys.
FUSED_MODALITIES = ("comp", "host", "func")

Validator = Literal["loocv", "rcv"]
Fusion = Literal["early", "late"]

VERSION_HELP = "feature version, a [features.vX_Y] block in config.toml"
SMOTE_HELP = "SMOTE oversampling inside each training fold"


def _check_validators(validators: Sequence[str]) -> list[str]:
    if not validators:
        return ["validators: give at least one of loocv, rcv"]
    if len(set(validators)) != len(validators):
        return [f"validators: duplicates in {validators}"]
    return []


def _feature_file(
    context: RunContext, version: str, modality: str, virus_identification_tool: str
) -> Path:
    return find_feature_file(
        context.feature_version_manager,
        version,
        context.modalities[modality],
        virus_identification_tool,
    )


def _feature_input(
    context: RunContext, version: str, modality: str, virus_identification_tool: str
) -> Path:
    """The feature file if it exists, else the directory an extraction job
    will write it into (so a classification after an extraction in the same
    run file is planned as depending on it)."""
    try:
        return _feature_file(context, version, modality, virus_identification_tool)
    except (FileNotFoundError, ValueError):
        run_config = context.modalities[modality]
        return context.stage_dir(version, "extracted") / run_config["tool"]


def _check_files(
    context: RunContext, version: str, modalities, virus_identification_tool: str
) -> list[str]:
    errors = []
    for modality in modalities:
        try:
            _feature_file(context, version, modality, virus_identification_tool)
        except FileNotFoundError as e:
            errors.append(str(e))
    return errors + check_metadata(context)


def _check_fused_modalities(context: RunContext) -> list[str]:
    missing = [modality for modality in FUSED_MODALITIES if modality not in context.modalities]
    if missing:
        return [
            f"[modalities] must define {', '.join(FUSED_MODALITIES)} (missing: {', '.join(missing)})"
        ]
    return []


# classification.single


@dataclass(frozen=True)
class SingleModalParams:
    version: str = param(help=VERSION_HELP)
    modality: str = param(help="a key of [modalities] in config.toml", grid=True)
    vtool: str = param(help=VIRUS_IDENTIFICATION_TOOL_HELP, grid=True)
    model: Model = param("catboost", help=MODEL_HELP, grid=True)
    validators: list[Validator] = param(default_factory=lambda: ["loocv"])
    smote: bool = param(False, help=SMOTE_HELP)
    seed: int = param(42, help=SEED_HELP, grid=True)
    out_dir: str | None = param(
        None, help="default: <[results]>/single-modal/<version>; relative = under the workspace"
    )


@register
class SingleModal(Stage):
    name = "classification.single"
    summary = "Single-modal classifier per modality x virus identification tool x model"
    Params = SingleModalParams

    def resolve(self, params, context: RunContext):
        return dataclasses.replace(
            params, vtool=resolve_virus_identification_tool(context, params.vtool)
        )

    def defaults(self, context: RunContext) -> dict:
        return {"vtool": context.virus_identification_tools}

    def choices(self, context: RunContext) -> dict[str, list]:
        return {
            "version": context.feature_versions(),
            "modality": list(context.modalities),
            "vtool": context.virus_identification_tools,
        }

    def validate(self, params: SingleModalParams, context: RunContext) -> list[str]:
        errors = check_version(params.version, context) + check_virus_identification_tool(
            params.vtool, context
        )
        if params.modality not in context.modalities:
            errors.append(
                f"modality {params.modality!r} not in [modalities] "
                f"(available: {', '.join(context.modalities)})"
            )
        return errors + _check_validators(params.validators)

    def check_inputs(self, params: SingleModalParams, context: RunContext) -> list[str]:
        return _check_files(context, params.version, [params.modality], params.vtool)

    def label(self, params: SingleModalParams) -> str:
        return f"{params.modality} / {params.vtool} / {params.model}"

    def inputs(self, params: SingleModalParams, context: RunContext) -> list[Path]:
        # The labels are an input too: relabeling must invalidate results.
        return [
            _feature_input(context, params.version, params.modality, params.vtool),
            metadata_path(context),
        ]

    def writes(self, params: SingleModalParams, context: RunContext) -> list[Path]:
        return [output_dir(params.out_dir, context, SINGLE_MODAL_DIR, params.version)]

    def run(self, params: SingleModalParams, context: RunContext) -> list[Path]:
        from phantom.analysis.classification.experiments.settings import SingleModalConfig
        from phantom.analysis.classification.experiments.single_modal import (
            SingleModalClassifier,
        )

        out_dir = output_dir(params.out_dir, context, SINGLE_MODAL_DIR, params.version)
        out_dir.mkdir(parents=True, exist_ok=True)
        classifier = SingleModalClassifier(
            SingleModalConfig(
                in_file=str(_feature_file(context, params.version, params.modality, params.vtool)),
                out_dir=str(out_dir),
                modality=params.modality,
                vtool=params.vtool,
                run_loocv="loocv" in params.validators,
                run_repeated="rcv" in params.validators,
                use_smote=params.smote,
                model_type=params.model,
                seed=params.seed,
            )
        )
        classifier.run()
        return [Path(classifier.saver.exp_dir)]


# classification.multi


@dataclass(frozen=True)
class MultiModalParams:
    version: str = param(help=VERSION_HELP)
    vtool: str = param(help=VIRUS_IDENTIFICATION_TOOL_HELP, grid=True)
    fusion: Fusion = param("early", grid=True)
    model: Model = param("catboost", help=MODEL_HELP, grid=True)
    validators: list[Validator] = param(default_factory=lambda: ["loocv"])
    smote: bool = param(False, help=SMOTE_HELP)
    optimize_weights: bool = param(
        False, help="late fusion only: tune per-modality weights with Optuna"
    )
    n_trials: int = param(30, help="Optuna trials when optimize_weights = true")
    seed: int = param(42, help=SEED_HELP, grid=True)
    out_dir: str | None = param(
        None, help="default: <[results]>/multi-modal/<version>; relative = under the workspace"
    )


@register
class MultiModal(Stage):
    name = "classification.multi"
    summary = "Multimodal (comp + host + func) early/late fusion classifier per vtool"
    Params = MultiModalParams

    def resolve(self, params, context: RunContext):
        return dataclasses.replace(
            params, vtool=resolve_virus_identification_tool(context, params.vtool)
        )

    def defaults(self, context: RunContext) -> dict:
        return {"vtool": context.virus_identification_tools}

    def choices(self, context: RunContext) -> dict[str, list]:
        return {"version": context.feature_versions(), "vtool": context.virus_identification_tools}

    def validate(self, params: MultiModalParams, context: RunContext) -> list[str]:
        errors = check_version(params.version, context) + check_virus_identification_tool(
            params.vtool, context
        )
        errors += _check_validators(params.validators)
        if params.optimize_weights and params.fusion != "late":
            errors.append("optimize_weights applies to fusion = 'late' only")
        if params.n_trials < 1:
            errors.append(f"n_trials must be >= 1, got {params.n_trials}")
        return errors + _check_fused_modalities(context)

    def check_inputs(self, params: MultiModalParams, context: RunContext) -> list[str]:
        return _check_files(context, params.version, FUSED_MODALITIES, params.vtool)

    def label(self, params: MultiModalParams) -> str:
        weights = " (tuned weights)" if params.optimize_weights else ""
        return f"{params.vtool} / {params.fusion}{weights} / {params.model}"

    def inputs(self, params: MultiModalParams, context: RunContext) -> list[Path]:
        files = [
            _feature_input(context, params.version, modality, params.vtool)
            for modality in FUSED_MODALITIES
        ]
        return [*files, metadata_path(context)]

    def writes(self, params: MultiModalParams, context: RunContext) -> list[Path]:
        return [output_dir(params.out_dir, context, MULTI_MODAL_DIR, params.version)]

    def run(self, params: MultiModalParams, context: RunContext) -> list[Path]:
        from phantom.analysis.classification.experiments.multi_modal import MultiModalClassifier
        from phantom.analysis.classification.experiments.settings import MultiModalConfig

        out_dir = output_dir(params.out_dir, context, MULTI_MODAL_DIR, params.version)
        out_dir.mkdir(parents=True, exist_ok=True)
        comp, host, func = (
            str(_feature_file(context, params.version, modality, params.vtool))
            for modality in FUSED_MODALITIES
        )
        # No weight tuning is (False, 0) for the executor.
        tune = params.fusion == "late" and params.optimize_weights
        classifier = MultiModalClassifier(
            MultiModalConfig(
                comp=comp,
                host=host,
                func=func,
                out_dir=str(out_dir),
                run_loocv="loocv" in params.validators,
                run_repeated="rcv" in params.validators,
                use_smote=params.smote,
                model_type=params.model,
                fusion=params.fusion,
                opt=tune,  # type: ignore[arg-type]  # the executor treats it as a flag
                n_trials=params.n_trials if tune else 0,
                vtool=params.vtool,
                seed=params.seed,
            )
        )
        classifier.run()
        return [Path(classifier.saver.exp_dir)]


# classification.mds


@dataclass(frozen=True)
class MdsParams:
    version: str = param(help=VERSION_HELP)
    vtool: str = param(help=VIRUS_IDENTIFICATION_TOOL_HELP, grid=True)
    seed: int = param(42, help=SEED_HELP, grid=True)
    out_dir: str | None = param(
        None, help="default: <[results]>/mds/<version>; relative = under the workspace"
    )


@register
class Mds(Stage):
    name = "classification.mds"
    summary = "Unsupervised MDS exploration of comp/host/func per vtool"
    Params = MdsParams

    def resolve(self, params, context: RunContext):
        return dataclasses.replace(
            params, vtool=resolve_virus_identification_tool(context, params.vtool)
        )

    def defaults(self, context: RunContext) -> dict:
        return {"vtool": context.virus_identification_tools}

    def choices(self, context: RunContext) -> dict[str, list]:
        return {"version": context.feature_versions(), "vtool": context.virus_identification_tools}

    def validate(self, params: MdsParams, context: RunContext) -> list[str]:
        return (
            check_version(params.version, context)
            + check_virus_identification_tool(params.vtool, context)
            + _check_fused_modalities(context)
        )

    def check_inputs(self, params: MdsParams, context: RunContext) -> list[str]:
        return _check_files(context, params.version, FUSED_MODALITIES, params.vtool)

    def label(self, params: MdsParams) -> str:
        return params.vtool

    def inputs(self, params: MdsParams, context: RunContext) -> list[Path]:
        files = [
            _feature_input(context, params.version, modality, params.vtool)
            for modality in FUSED_MODALITIES
        ]
        return [*files, metadata_path(context)]

    def writes(self, params: MdsParams, context: RunContext) -> list[Path]:
        return [output_dir(params.out_dir, context, MDS_DIR, params.version)]

    def run(self, params: MdsParams, context: RunContext) -> list[Path]:
        from phantom.analysis.classification.experiments.mds import (
            MdsAnalysis,
        )
        from phantom.analysis.classification.experiments.settings import MdsConfig

        out_dir = output_dir(params.out_dir, context, MDS_DIR, params.version)
        out_dir.mkdir(parents=True, exist_ok=True)
        comp, host, func = (
            str(_feature_file(context, params.version, modality, params.vtool))
            for modality in FUSED_MODALITIES
        )
        config = MdsConfig(comp=comp, host=host, func=func, out_dir=str(out_dir), seed=params.seed)
        classifier = MdsAnalysis(config)
        classifier.run()
        return [Path(classifier.saver.exp_dir)]


# classification.figures

FIGURES_DIR = "figures"
Kind = Literal["single", "multi"]


@dataclass(frozen=True)
class FiguresParams:
    version: str = param(help=VERSION_HELP)
    model: Model = param("catboost", help=MODEL_HELP, grid=True)
    validators: list[Validator] = param(default_factory=lambda: ["loocv", "rcv"])
    kinds: list[Kind] = param(
        default_factory=lambda: ["single", "multi"],
        help="single = classification.single runs, multi = classification.multi runs",
    )
    out_dir: str | None = param(
        None,
        help="default: <[results]>/figures/<version>/<model>; relative = under the workspace",
    )


@register
class Figures(Stage):
    name = "classification.figures"
    summary = "Figures + metrics table of an experiment: metric grids, confusion matrices, misclassified samples, top features"
    Params = FiguresParams

    def choices(self, context: RunContext) -> dict[str, list]:
        return {"version": context.feature_versions()}

    def validate(self, params: FiguresParams, context: RunContext) -> list[str]:
        errors = check_version(params.version, context) + _check_validators(params.validators)
        if not params.kinds:
            errors.append("kinds: give at least one of single, multi")
        return errors

    def _results(self, params: FiguresParams, context: RunContext) -> dict[str, Path]:
        dirs = {"single": SINGLE_MODAL_DIR, "multi": MULTI_MODAL_DIR}
        return {k: output_dir(None, context, dirs[k], params.version) for k in params.kinds}

    def check_inputs(self, params: FiguresParams, context: RunContext) -> list[str]:
        from phantom.analysis.classification.reporting.experiment_figures import collect_runs

        results = self._results(params, context)
        if any(collect_runs(path, kind, params.model) for kind, path in results.items()):
            return []
        places = ", ".join(str(path) for path in results.values())
        return [f"no {params.model} classification runs in {places} -- run classification.* first"]

    def label(self, params: FiguresParams) -> str:
        return f"{params.model} / {' + '.join(params.kinds)}"

    def inputs(self, params: FiguresParams, context: RunContext) -> list[Path]:
        return list(self._results(params, context).values())

    def writes(self, params: FiguresParams, context: RunContext) -> list[Path]:
        return [output_dir(params.out_dir, context, FIGURES_DIR, params.version, params.model)]

    def run(self, params: FiguresParams, context: RunContext) -> list[Path]:
        from phantom.analysis.classification.reporting.experiment_figures import (
            Labels,
            make_figures,
        )

        out_dir = output_dir(params.out_dir, context, FIGURES_DIR, params.version, params.model)
        written = make_figures(
            self._results(params, context),
            params.model,
            list(params.validators),
            Labels.from_config(context.config),
            out_dir,
        )
        print(f"[INFO] {len(written)} file(s) in {out_dir}")
        return [out_dir]
