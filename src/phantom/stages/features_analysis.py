"""
Analyses of a feature version (not part of the feature-building chain).

``features.diversity``            taxonomic/host diversity, cases vs. controls
``features.summarize_functions``  per-category FASTA + genus/host table of ORFs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from phantom.runner.parameters import param
from phantom.runner.stage import RunContext, Stage, register
from phantom.stages._common import check_metadata, check_version, metadata_path, output_dir

VERSION_HELP = "feature version, a [features.vX_Y] block in config.toml"
DIVERSITY_DIR = "diversity"
# Tools the diversity analysis reads (DiversityAnalyzer.TOOL_FEATURE_COL).
DIVERSITY_TOOLS = ("phagcn", "cherry")
# Tools whose preprocessed tables annotate ORFs (FunctionSummarizer).
SUMMARY_TOOLS = ("phavip", "phagcn", "cherry")


def _preprocessed(context: RunContext, version: str) -> Path:
    return context.stage_dir(version, "preprocessed")


# features.diversity


@dataclass(frozen=True)
class DiversityParams:
    version: str = param(help=VERSION_HELP)
    ratio_mode: Literal["ratio_of_means", "mean_of_ratios"] = param(
        "ratio_of_means",
        grid=True,
        help="per-genome metric: mean(units)/mean(genomes) per group, or mean(units/genomes)",
    )
    out_dir: str | None = param(
        None, help="default: <[results]>/diversity/<version>; relative = under the workspace"
    )


@register
class Diversity(Stage):
    name = "features.diversity"
    summary = "Phage-genus and bacterial-host diversity, cases vs. controls"
    Params = DiversityParams

    def choices(self, context: RunContext) -> dict[str, list]:
        return {"version": context.feature_versions()}

    def validate(self, params: DiversityParams, context: RunContext) -> list[str]:
        return check_version(params.version, context)

    def check_inputs(self, params: DiversityParams, context: RunContext) -> list[str]:
        pre = _preprocessed(context, params.version)
        errors = []
        if not any((pre / t).is_dir() for t in DIVERSITY_TOOLS):
            errors.append(f"no preprocessed phagcn/cherry data under {pre}")
        return errors + check_metadata(context)

    def label(self, params: DiversityParams) -> str:
        return params.ratio_mode

    def inputs(self, params: DiversityParams, context: RunContext) -> list[Path]:
        pre = _preprocessed(context, params.version)
        return [*(pre / t for t in DIVERSITY_TOOLS), metadata_path(context)]

    def writes(self, params: DiversityParams, context: RunContext) -> list[Path]:
        return [output_dir(params.out_dir, context, DIVERSITY_DIR, params.version)]

    def run(self, params: DiversityParams, context: RunContext) -> list[Path]:
        from phantom.analysis.features.diversity import DiversityAnalyzer

        out_dir = output_dir(params.out_dir, context, DIVERSITY_DIR, params.version)
        DiversityAnalyzer(
            version=params.version, feature_version_manager=context.feature_version_manager
        ).run(out_dir, ratio_mode=params.ratio_mode)
        return [out_dir]


# features.summarize_functions


@dataclass(frozen=True)
class SummarizeParams:
    version: str = param(help=VERSION_HELP)


@register
class SummarizeFunctions(Stage):
    name = "features.summarize_functions"
    summary = "Group PHANOTATE ORFs by phavip function, with genus (phagcn) and host (cherry)"
    Params = SummarizeParams

    def choices(self, context: RunContext) -> dict[str, list]:
        return {"version": context.feature_versions()}

    def validate(self, params: SummarizeParams, context: RunContext) -> list[str]:
        errors = check_version(params.version, context)
        for section in ("phanotate", "functions"):
            if not context.config.get(section, {}).get("path"):
                errors.append(f"config.toml is missing [{section}].path")
        return errors

    def check_inputs(self, params: SummarizeParams, context: RunContext) -> list[str]:
        errors = []
        if not (_preprocessed(context, params.version) / "phavip").is_dir():
            errors.append(
                f"no preprocessed phavip data under {_preprocessed(context, params.version)}"
            )
        if not context.section_path("phanotate").is_dir():
            errors.append(f"PHANOTATE ORF directory not found: {context.section_path('phanotate')}")
        return errors

    def inputs(self, params: SummarizeParams, context: RunContext) -> list[Path]:
        pre = _preprocessed(context, params.version)
        return [*(pre / t for t in SUMMARY_TOOLS), context.section_path("phanotate")]

    def writes(self, params: SummarizeParams, context: RunContext) -> list[Path]:
        return [context.section_path("functions") / params.version]

    def run(self, params: SummarizeParams, context: RunContext) -> list[Path]:
        from phantom.analysis.features.function_summary import FunctionSummarizer

        summarizer = FunctionSummarizer(
            params.version, context.config, context.feature_version_manager
        )
        summarizer.run()
        return [summarizer.out_root]
