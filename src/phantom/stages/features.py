"""
Feature stages -- replace the interactive Features menu.

``features.diversity``            healthy-vs-allergy taxonomic/host diversity
``features.summarize_functions``  per-category FASTA + genus/host table of ORFs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from phantom.runner.params import param
from phantom.runner.stage import RunContext, Stage, register
from phantom.stages._common import check_metadata, check_version, metadata_path, output_dir

VERSION_HELP = "feature version, a [features.vX_Y] block in config.toml"
DIVERSITY_DIR = "diversity"
# Tools the diversity analysis reads (DiversityAnalyzer.TOOL_FEATURE_COL).
DIVERSITY_TOOLS = ("phagcn", "cherry")
# Tools whose preprocessed tables annotate ORFs (FunctionSummarizer).
SUMMARY_TOOLS = ("phavip", "phagcn", "cherry")


def _preprocessed(ctx: RunContext, version: str) -> Path:
    return ctx.stage_dir(version, "preprocessed")


# --- features.diversity -------------------------------------------------------


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
    summary = "Phage-genus and bacterial-host diversity, healthy vs. allergy"
    Params = DiversityParams

    def choices(self, ctx: RunContext) -> dict[str, list]:
        return {"version": ctx.feature_versions()}

    def validate(self, p: DiversityParams, ctx: RunContext) -> list[str]:
        return check_version(p.version, ctx)

    def check_inputs(self, p: DiversityParams, ctx: RunContext) -> list[str]:
        pre = _preprocessed(ctx, p.version)
        errors = []
        if not any((pre / t).is_dir() for t in DIVERSITY_TOOLS):
            errors.append(f"no preprocessed phagcn/cherry data under {pre}")
        return errors + check_metadata(ctx)

    def label(self, p: DiversityParams) -> str:
        return p.ratio_mode

    def inputs(self, p: DiversityParams, ctx: RunContext) -> list[Path]:
        pre = _preprocessed(ctx, p.version)
        return [*(pre / t for t in DIVERSITY_TOOLS), metadata_path(ctx)]

    def writes(self, p: DiversityParams, ctx: RunContext) -> list[Path]:
        return [output_dir(p.out_dir, ctx, DIVERSITY_DIR, p.version)]

    def run(self, p: DiversityParams, ctx: RunContext) -> list[Path]:
        from phantom.features.analytics.diversity import DiversityAnalyzer

        out_dir = output_dir(p.out_dir, ctx, DIVERSITY_DIR, p.version)
        DiversityAnalyzer(version=p.version, config_mgr=ctx.config_mgr).run(
            out_dir, ratio_mode=p.ratio_mode
        )
        return [out_dir]


# --- features.summarize_functions ---------------------------------------------


@dataclass(frozen=True)
class SummarizeParams:
    version: str = param(help=VERSION_HELP)


@register
class SummarizeFunctions(Stage):
    name = "features.summarize_functions"
    summary = "Group PHANOTATE ORFs by phavip function, with genus (phagcn) and host (cherry)"
    Params = SummarizeParams

    def choices(self, ctx: RunContext) -> dict[str, list]:
        return {"version": ctx.feature_versions()}

    def validate(self, p: SummarizeParams, ctx: RunContext) -> list[str]:
        errors = check_version(p.version, ctx)
        for section in ("phanotate", "functions"):
            if not ctx.config.get(section, {}).get("path"):
                errors.append(f"config.toml is missing [{section}].path")
        return errors

    def check_inputs(self, p: SummarizeParams, ctx: RunContext) -> list[str]:
        errors = []
        if not (_preprocessed(ctx, p.version) / "phavip").is_dir():
            errors.append(f"no preprocessed phavip data under {_preprocessed(ctx, p.version)}")
        if not ctx.section_path("phanotate").is_dir():
            errors.append(f"PHANOTATE ORF directory not found: {ctx.section_path('phanotate')}")
        return errors

    def inputs(self, p: SummarizeParams, ctx: RunContext) -> list[Path]:
        pre = _preprocessed(ctx, p.version)
        return [*(pre / t for t in SUMMARY_TOOLS), ctx.section_path("phanotate")]

    def writes(self, p: SummarizeParams, ctx: RunContext) -> list[Path]:
        return [ctx.section_path("functions") / p.version]

    def run(self, p: SummarizeParams, ctx: RunContext) -> list[Path]:
        from phantom.features.summary.function_summarizer import FunctionSummarizer

        summarizer = FunctionSummarizer(version=p.version, config_mgr=ctx.config_mgr)
        summarizer.run()
        return [summarizer.out_root]
