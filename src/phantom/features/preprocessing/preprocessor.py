"""
Preprocesses merged raw feature files per tool: applies each tool's
domain-specific cleaning/filtering before the final feature-matrix
extraction step.

Which CheckV mask applies to a file is decided by ``mask_for(merged_file)``
-- a fixed choice from a run file (:func:`named_mask`), or, when omitted, the
legacy interactive prompt.
"""

from collections.abc import Callable
from pathlib import Path

from phantom.config.features import FeatureConfigManager
from phantom.config.loader import ConfigLoader
from phantom.features.pipelines.tools.cherry import CherryFeaturePipeline
from phantom.features.pipelines.tools.phagcn import PhagcnFeaturePipeline
from phantom.features.pipelines.tools.phatyp import PhatypFeaturePipeline
from phantom.features.pipelines.tools.phavip import PhavipFeaturePipeline
from phantom.features.pipelines.utils import apply_mask, load_file

PIPELINES = {
    "cherry": CherryFeaturePipeline,
    "phagcn": PhagcnFeaturePipeline,
    "phavip": PhavipFeaturePipeline,
    "phatyp": PhatypFeaturePipeline,
}

MaskChooser = Callable[[Path], Path | None]


def mask_dir_for(merged_file: Path, config: dict) -> Path:
    """
    The per-vtool mask directory for a merged file: ``<[masks].path>/<tool>``
    where ``<tool>`` is the ``[tools]`` name of the file's prefix
    (``geN_CHR_M.csv`` -> ``genomad``).
    """
    masks = config.get("masks", {}).get("path")
    if not masks:
        raise ValueError("config.toml is missing [masks].path")
    prefix = merged_file.stem.split("_")[0]
    tool_name = config.get("tools", {}).get(prefix)
    if tool_name is None:
        raise ValueError(f"unknown vtool prefix {prefix!r} of {merged_file.name} (see [tools])")
    return ConfigLoader.resolve_data_path(masks) / tool_name


def named_mask(config: dict, mask_name: str | None) -> MaskChooser:
    """The same mask file (e.g. ``MQ_vMAGs.tsv``) from each file's vtool directory."""

    def choose(merged_file: Path) -> Path | None:
        return mask_dir_for(merged_file, config) / mask_name if mask_name else None

    return choose


class FeaturePreprocessor:
    def __init__(
        self,
        version: str,
        config_mgr: FeatureConfigManager | None = None,
        mask_for: MaskChooser | None = None,
        tools: list[str] | None = None,
    ):
        self.version = version
        self.config_mgr = config_mgr or FeatureConfigManager()
        self.raw_merged_dir = self.config_mgr.get_stage_dir(version, "raw_merged")
        self.preprocessed_dir = self.config_mgr.get_stage_dir(version, "preprocessed")
        self.tools = list(PIPELINES) if tools is None else tools
        if mask_for is None:  # legacy interactive menu
            from phantom.cli.features import FeatureExtractionPrompts

            mask_for = FeatureExtractionPrompts().ask_mask_file
        self.mask_for = mask_for

    def run(self) -> list[Path]:
        """Preprocesses every merged file of the selected tools; returns the
        per-tool output directories written."""
        print(f"\n[INFO] Starting feature preprocessing for version {self.version}...")
        if not self.raw_merged_dir.exists():
            print(f"[ERROR] Input directory does not exist: {self.raw_merged_dir}")
            return []
        written = []
        for tool, pipeline_cls in PIPELINES.items():
            tool_dir = self.raw_merged_dir / tool
            if tool not in self.tools or not tool_dir.is_dir():
                continue
            for merged_file in sorted(tool_dir.glob("*.csv")):
                self._process_file(tool, pipeline_cls(), merged_file)
            written.append(self.preprocessed_dir / tool)
        if not written:
            print(
                f"[ERROR] No recognized tool directories ({', '.join(self.tools)}) "
                f"found under {self.raw_merged_dir}"
            )
        return written

    def _process_file(self, tool: str, pipeline, merged_file: Path) -> None:
        print(f"\n[INFO] Preprocessing {tool}/{merged_file.name}...")
        df = load_file(merged_file)
        mask_path = self.mask_for(merged_file)
        df = apply_mask(df, mask_path)
        preprocessed_df = pipeline.preprocess(df)
        out_dir = self.preprocessed_dir / tool
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{merged_file.stem}_PP.csv"
        preprocessed_df.to_csv(out_path, sep=";", index=False)
        print(f"       Saved to: {out_path} (Shape: {preprocessed_df.shape})")
