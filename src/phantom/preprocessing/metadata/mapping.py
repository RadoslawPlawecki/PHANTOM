"""
Manages the metadata file that stores information about samples and
dataset statistics.
"""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from phantom.config.loader import ConfigLoader


class MetadataManager:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def path_from_config(config: dict) -> Path:
        """
        Resolves the merged metadata CSV's location from config.toml's
        ``[genomes].metadata_path`` -- the one place both preprocessing
        (which writes it) and classification (which reads its ``label``
        column) agree on where it lives. There is deliberately no
        fallback path in code: config.toml is the single source of truth.
        """
        rel_path = config.get("genomes", {}).get("metadata_path")
        if not rel_path:
            raise ValueError(
                "config.toml is missing [genomes].metadata_path "
                '(e.g. metadata_path = "data/genomes/genome_metadata.csv").'
            )
        return ConfigLoader.resolve_data_path(rel_path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["id"])
        return pd.read_csv(self.path, sep=";")

    def save(self, df: pd.DataFrame):
        df.to_csv(self.path, sep=";", index=False)

    def add_metadata_columns(self, new_df: pd.DataFrame, overwrite: Iterable[str] = ()):
        """
        Merges new metadata on 'id'. Fails if duplicate columns (other
        than 'id') exist, unless they are explicitly listed in
        ``overwrite`` -- those are dropped from the current file first and
        replaced wholesale by ``new_df``'s values.
        """
        if new_df.empty or "id" not in new_df.columns:
            return
        current_df = self.load()
        if current_df.empty:
            self.save(new_df)
            return
        overwrite = set(overwrite) - {"id"}
        current_df = current_df.drop(columns=list(overwrite & set(current_df.columns)))
        duplicate_cols = set(new_df.columns).intersection(current_df.columns) - {"id"}
        if duplicate_cols:
            raise ValueError(
                f"Duplicate columns found: {duplicate_cols}. Incrementing not allowed."
            )
        merged = pd.merge(current_df, new_df, on="id", how="outer")
        if "id" in merged.columns:
            merged["_sort_key"] = merged["id"].str.extract(r"(\d+)", expand=False).astype(float)
            merged = (
                merged.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
            )
        self.save(merged)
