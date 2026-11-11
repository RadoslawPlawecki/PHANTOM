"""
Builds final feature matrices from preprocessed per-tool feature data.

The per-file matrix options (feature column, binary vs. counts, minimum
patients) come from ``options_for(tool, pipeline, df)`` -- fixed values from
a run file (:func:`fixed_options`), or, when omitted, the legacy interactive
prompts.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from phantom.config.features import FeatureConfigManager
from phantom.features.pipelines.tools.cherry import CherryFeaturePipeline
from phantom.features.pipelines.tools.phagcn import PhagcnFeaturePipeline
from phantom.features.pipelines.tools.phatyp import PhatypFeaturePipeline
from phantom.features.pipelines.tools.phavip import PhavipFeaturePipeline

PIPELINES = {
    "cherry": CherryFeaturePipeline,
    "phagcn": PhagcnFeaturePipeline,
    "phavip": PhavipFeaturePipeline,
    "phatyp": PhatypFeaturePipeline,
}


@dataclass(frozen=True)
class MatrixOptions:
    feature_col: str | None = None
    binary: bool | None = None  # None = the pipeline's own default
    min_patients: int | None = None  # None = the pipeline's own default


OptionsChooser = Callable[[str, object, pd.DataFrame], MatrixOptions]


def _per_tool(value, tool: str):
    return value.get(tool) if isinstance(value, dict) else value


def fixed_options(
    feature_col: dict[str, str] | None = None,
    binary: bool | dict[str, bool] | None = None,
    min_patients: int | dict[str, int] | None = None,
) -> OptionsChooser:
    """Options from a run file: per-tool tables, or one value for every tool."""

    def choose(tool: str, pipeline, df: pd.DataFrame) -> MatrixOptions:
        if not pipeline.NEEDS_MATRIX_OPTIONS and not pipeline.NEEDS_FEATURE_COLUMN:
            return MatrixOptions()
        return MatrixOptions(
            feature_col=(feature_col or {}).get(tool) if pipeline.NEEDS_FEATURE_COLUMN else None,
            binary=_per_tool(binary, tool) if pipeline.NEEDS_MATRIX_OPTIONS else None,
            min_patients=_per_tool(min_patients, tool) if pipeline.NEEDS_MATRIX_OPTIONS else None,
        )

    return choose


def _prompted_options() -> OptionsChooser:
    """The legacy interactive menu: ask per file."""
    from phantom.cli.features import FeatureExtractionPrompts

    prompts = FeatureExtractionPrompts()

    def choose(tool: str, pipeline, df: pd.DataFrame) -> MatrixOptions:
        feature_col = prompts.ask_column(df) if pipeline.NEEDS_FEATURE_COLUMN else None
        if not pipeline.NEEDS_MATRIX_OPTIONS:
            return MatrixOptions(feature_col=feature_col)
        return MatrixOptions(
            feature_col=feature_col,
            binary=prompts.ask_binary(default=pipeline.binary),
            min_patients=prompts.ask_min_patients(default=pipeline.min_patients),
        )

    return choose


class FeatureExtractor:
    def __init__(
        self,
        version: str,
        config_mgr: FeatureConfigManager | None = None,
        options_for: OptionsChooser | None = None,
        tools: list[str] | None = None,
    ):
        self.version = version
        self.config_mgr = config_mgr or FeatureConfigManager()
        self.preprocessed_dir = self.config_mgr.get_stage_dir(version, "preprocessed")
        self.extracted_dir = self.config_mgr.get_stage_dir(version, "extracted")
        self.tools = list(PIPELINES) if tools is None else tools
        self.options_for = options_for or _prompted_options()

    def run(self) -> list[Path]:
        """Extracts every preprocessed file of the selected tools; returns
        the per-tool output directories written."""
        print(f"\n[INFO] Starting feature extraction for version {self.version}...")
        if not self.preprocessed_dir.exists():
            print(f"[ERROR] Input directory does not exist: {self.preprocessed_dir}")
            return []
        written = []
        for tool, pipeline_cls in PIPELINES.items():
            tool_dir = self.preprocessed_dir / tool
            if tool not in self.tools or not tool_dir.is_dir():
                continue
            for preprocessed_file in sorted(tool_dir.glob("*.csv")):
                self._process_file(tool, pipeline_cls(), preprocessed_file)
            written.append(self.extracted_dir / tool)
        if not written:
            print(
                f"[ERROR] No preprocessed data found under {self.preprocessed_dir}. "
                f"Run preprocessing first."
            )
        return written

    def _process_file(self, tool: str, pipeline, preprocessed_file: Path) -> None:
        print(f"\n[INFO] Extracting features for {tool}/{preprocessed_file.name}...")
        df = pd.read_csv(preprocessed_file, sep=";")
        opts = self.options_for(tool, pipeline, df)
        feature_matrix = pipeline.build_feature_matrix(
            df, opts.feature_col, binary=opts.binary, min_patients=opts.min_patients
        )
        out_dir = self.extracted_dir / tool
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = preprocessed_file.stem
        if stem.endswith("_M_PP"):
            stem = stem[: -len("_M_PP")]
        out_path = out_dir / f"{stem}_FEAT.csv"
        feature_matrix.to_csv(out_path, sep=";", index=False)
        print(f"       Saved to: {out_path} (Shape: {feature_matrix.shape})")
