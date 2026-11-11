"""
Preprocesses merged raw feature files per tool: applies each tool's
domain-specific cleaning/filtering before the final feature-matrix
extraction step, optionally keeping only contigs in a CheckV mask.
"""

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


class FeaturePreprocessor:
    def __init__(
        self,
        version: str,
        config: dict,
        config_mgr: FeatureConfigManager,
        mask: str | None = None,
        tools: list[str] | None = None,
    ):
        """
        ``mask``: a CheckV mask file name (e.g. ``MQ_vMAGs.tsv``) taken from
        each merged file's vtool directory; None = no masking. ``tools``:
        which tool directories to process; None = all of them.
        """
        self.version = version
        self.config = config
        self.mask = mask
        self.tools = list(PIPELINES) if tools is None else tools
        self.raw_merged_dir = config_mgr.get_stage_dir(version, "raw_merged")
        self.preprocessed_dir = config_mgr.get_stage_dir(version, "preprocessed")

    def run(self) -> list[Path]:
        """Preprocesses every merged file of the selected tools; returns the
        per-tool output directories written."""
        print(f"\n[INFO] Starting feature preprocessing for version {self.version}...")
        written = []
        for tool, pipeline_cls in PIPELINES.items():
            tool_dir = self.raw_merged_dir / tool
            if tool not in self.tools or not tool_dir.is_dir():
                continue
            for merged_file in sorted(tool_dir.glob("*.csv")):
                self._process_file(tool, pipeline_cls(), merged_file)
            written.append(self.preprocessed_dir / tool)
        return written

    def _process_file(self, tool: str, pipeline, merged_file: Path) -> None:
        print(f"\n[INFO] Preprocessing {tool}/{merged_file.name}...")
        df = load_file(merged_file)
        mask_path = mask_dir_for(merged_file, self.config) / self.mask if self.mask else None
        df = apply_mask(df, mask_path)
        preprocessed_df = pipeline.preprocess(df)
        out_dir = self.preprocessed_dir / tool
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{merged_file.stem}_PP.csv"
        preprocessed_df.to_csv(out_path, sep=";", index=False)
        print(f"       Saved to: {out_path} (Shape: {preprocessed_df.shape})")
