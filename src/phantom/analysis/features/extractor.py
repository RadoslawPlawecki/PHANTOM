"""
Builds final feature matrices from preprocessed per-tool feature data.
"""

from pathlib import Path

import pandas as pd

from phantom.analysis.features.phabox2.cherry import CherryFeaturePipeline
from phantom.analysis.features.phabox2.phagcn import PhagcnFeaturePipeline
from phantom.analysis.features.phabox2.phatyp import PhatypFeaturePipeline
from phantom.analysis.features.phabox2.phavip import PhavipFeaturePipeline
from phantom.config.feature_versions import FeatureConfigManager

PIPELINES = {
    "cherry": CherryFeaturePipeline,
    "phagcn": PhagcnFeaturePipeline,
    "phavip": PhavipFeaturePipeline,
    "phatyp": PhatypFeaturePipeline,
}


def _per_tool(value, tool: str):
    """A run-file option given once for every tool, or as a {tool: value} table."""
    return value.get(tool) if isinstance(value, dict) else value


class FeatureExtractor:
    def __init__(
        self,
        version: str,
        config_mgr: FeatureConfigManager,
        feature_col: dict[str, str] | None = None,
        binary: bool | dict[str, bool] | None = None,
        min_patients: int | dict[str, int] | None = None,
        tools: list[str] | None = None,
    ):
        """
        ``feature_col``: the taxonomy column per tool that needs one
        (``NEEDS_FEATURE_COLUMN``). ``binary`` / ``min_patients``: for tools
        with ``NEEDS_MATRIX_OPTIONS``; one value or one per tool; None = the
        pipeline's own default. Tools that need none of them ignore them.
        """
        self.version = version
        self.feature_col = feature_col or {}
        self.binary = binary
        self.min_patients = min_patients
        self.tools = list(PIPELINES) if tools is None else tools
        self.preprocessed_dir = config_mgr.get_stage_dir(version, "preprocessed")
        self.extracted_dir = config_mgr.get_stage_dir(version, "extracted")

    def run(self) -> list[Path]:
        """Extracts every preprocessed file of the selected tools; returns
        the per-tool output directories written."""
        print(f"\n[INFO] Starting feature extraction for version {self.version}...")
        written = []
        for tool, pipeline_cls in PIPELINES.items():
            tool_dir = self.preprocessed_dir / tool
            if tool not in self.tools or not tool_dir.is_dir():
                continue
            for preprocessed_file in sorted(tool_dir.glob("*.csv")):
                self._process_file(tool, pipeline_cls(), preprocessed_file)
            written.append(self.extracted_dir / tool)
        return written

    def _process_file(self, tool: str, pipeline, preprocessed_file: Path) -> None:
        print(f"\n[INFO] Extracting features for {tool}/{preprocessed_file.name}...")
        df = pd.read_csv(preprocessed_file, sep=";")
        options = pipeline.NEEDS_MATRIX_OPTIONS
        feature_matrix = pipeline.build_feature_matrix(
            df,
            self.feature_col.get(tool) if pipeline.NEEDS_FEATURE_COLUMN else None,
            binary=_per_tool(self.binary, tool) if options else None,
            min_patients=_per_tool(self.min_patients, tool) if options else None,
        )
        out_dir = self.extracted_dir / tool
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = preprocessed_file.stem.removesuffix("_M_PP")
        out_path = out_dir / f"{stem}_FEAT.csv"
        feature_matrix.to_csv(out_path, sep=";", index=False)
        print(f"       Saved to: {out_path} (Shape: {feature_matrix.shape})")
