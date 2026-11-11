"""
Classification stages.

``classification.single`` -- one supervised classifier per (modality,
virus identification tool, model) job, on a feature version's extracted
matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from phantom.classification.data.inputs import find_feature_file
from phantom.preprocessing.metadata.mapping import MetadataManager
from phantom.runner.params import param
from phantom.runner.stage import RunContext, Stage, register

# Subdirectory of [results] per run type.
SINGLE_OMIC_DIR = "single-omic"

Model = Literal["catboost", "rf", "xgb"]
Validator = Literal["loocv", "rcv"]


@dataclass(frozen=True)
class SingleOmicParams:
    version: str = param(help="feature version, a [features.vX_Y] block in config.toml")
    modality: str = param(help="a key of [modalities] in config.toml", grid=True)
    vtool: str = param(help="a key of [tools] in config.toml", grid=True)
    model: Model = param("catboost", grid=True)
    validators: list[Validator] = param(default_factory=lambda: ["loocv"])
    smote: bool = param(False, help="SMOTE oversampling inside each training fold")
    out_dir: str | None = param(
        None, help="default: <[results]>/single-omic/<version>; relative = under the workspace"
    )


@register
class SingleOmic(Stage):
    name = "classification.single"
    summary = "Single-omic classifier per modality x virus-identification tool x model"
    Params = SingleOmicParams

    def defaults(self, ctx: RunContext) -> dict:
        return {"vtool": ctx.vtools}

    def choices(self, ctx: RunContext) -> dict[str, list]:
        return {
            "version": ctx.feature_versions(),
            "modality": list(ctx.modalities),
            "vtool": ctx.vtools,
        }

    def validate(self, p: SingleOmicParams, ctx: RunContext) -> list[str]:
        errors = []
        versions = ctx.feature_versions()
        if p.version not in versions:
            errors.append(
                f"version {p.version!r} not in config.toml (available: {', '.join(versions)})"
            )
        if p.modality not in ctx.modalities:
            errors.append(
                f"modality {p.modality!r} not in [modalities] (available: {', '.join(ctx.modalities)})"
            )
        if p.vtool not in ctx.vtools:
            errors.append(f"vtool {p.vtool!r} not in [tools] (available: {', '.join(ctx.vtools)})")
        if not p.validators:
            errors.append("validators: give at least one of loocv, rcv")
        elif len(set(p.validators)) != len(p.validators):
            errors.append(f"validators: duplicates in {p.validators}")
        if not errors:
            try:
                self._in_file(p, ctx)
            except FileNotFoundError as e:
                errors.append(str(e))
            labels = MetadataManager.path_from_config(ctx.config)
            if not labels.is_file():
                errors.append(f"metadata with class labels not found: {labels}")
        return errors

    def label(self, p: SingleOmicParams) -> str:
        return f"{p.modality} / {p.vtool} / {p.model}"

    def _in_file(self, p: SingleOmicParams, ctx: RunContext) -> Path:
        return find_feature_file(ctx.config_mgr, p.version, ctx.modalities[p.modality], p.vtool)

    def _out_dir(self, p: SingleOmicParams, ctx: RunContext) -> Path:
        if p.out_dir:
            return ctx.resolve(p.out_dir)
        return ctx.section_path("results") / SINGLE_OMIC_DIR / p.version

    def inputs(self, p: SingleOmicParams, ctx: RunContext) -> list[Path]:
        # The labels are an input too: relabeling must invalidate results.
        return [self._in_file(p, ctx), MetadataManager.path_from_config(ctx.config)]

    def writes(self, p: SingleOmicParams, ctx: RunContext) -> list[Path]:
        return [self._out_dir(p, ctx)]

    def run(self, p: SingleOmicParams, ctx: RunContext) -> list[Path]:
        from phantom.classification.data.config import SingleOmicConfig
        from phantom.classification.execution.single_omic import SingleOmicClassifier

        out_dir = self._out_dir(p, ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        classifier = SingleOmicClassifier(
            SingleOmicConfig(
                in_file=str(self._in_file(p, ctx)),
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
