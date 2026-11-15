"""
Merges raw per-sample feature files (TSV/CSV) into one file per tool and
virus identification tool. Each sample's sequence ids are prefixed with the
sample id (``S1_`` + id) and the id column is renamed ``Accession``.
"""

import fnmatch
import os
from pathlib import Path

import pandas as pd

from phantom.config.feature_versions import FeatureConfigManager


class FeatureMerger:
    def __init__(
        self,
        version: str,
        feature_version_manager: FeatureConfigManager,
        id_columns: dict[str, str],
    ):
        """
        ``id_columns``: ``{"phavip/*": "Genome", "*": "Accession"}`` -- the
        first glob pattern (in table order) matching a directory's path
        relative to ``raw/`` names its sequence-id column.
        """
        self.version = version
        self.id_columns = id_columns
        self.raw_dir = feature_version_manager.get_stage_dir(version, "raw")
        self.raw_merged_dir = feature_version_manager.get_stage_dir(version, "raw_merged")
        self.written: list[Path] = []

    def id_column_for(self, folder_label: str) -> str | None:
        for pattern, column in self.id_columns.items():
            if fnmatch.fnmatch(folder_label, pattern):
                return column
        return None

    def run(self) -> list[Path]:
        """Merges every directory holding raw files; returns the files written."""
        print(f"\n[INFO] Starting feature merging for version {self.version}...")
        for leaf_dir in self.leaf_dirs():
            self._process_leaf_directory(leaf_dir)
        return self.written

    def leaf_dirs(self) -> list[Path]:
        return self._find_leaf_dirs(self.raw_dir) if self.raw_dir.exists() else []

    def _find_leaf_dirs(self, root: Path) -> list[Path]:
        """
        Recursively finds every directory that directly contains .tsv/.csv
        files, regardless of how deeply it is nested under `root`. This
        makes merging independent of the exact directory depth produced
        during import (a single "<tool>/<module>" level is the common
        case, but a directory of raw files at any depth must still merge).
        """
        leaf_dirs = []
        for dirpath, _, filenames in os.walk(root):
            if any(f.endswith((".tsv", ".csv")) for f in filenames):
                leaf_dirs.append(Path(dirpath))
        return sorted(leaf_dirs)

    def _process_leaf_directory(self, leaf_dir: Path) -> None:
        raw_files = self.raw_files(leaf_dir)
        if not raw_files:
            return
        rel_dir = leaf_dir.relative_to(self.raw_dir)
        folder_label = str(rel_dir)
        id_col = self.id_column_for(folder_label)
        if not id_col:
            print(f"[ERROR] Skipped {folder_label}: No ID column selected.")
            return
        dfs = []
        sample_name = None
        for file in raw_files:
            df, s_name = self._read_and_format_file(file, id_col)
            if df is not None:
                dfs.append(df)
                if sample_name is None:
                    sample_name = s_name
        if dfs:
            self._save_merged_df(dfs, rel_dir, sample_name)

    @staticmethod
    def peek_columns(sample_file: Path) -> list[str]:
        sep = "\t" if sample_file.suffix.lower() == ".tsv" else ";"
        try:
            peek_df = pd.read_csv(sample_file, sep=sep, nrows=2)
        except Exception:
            peek_df = pd.read_csv(sample_file, sep=None, engine="python", nrows=2)
        return list(peek_df.columns)

    @staticmethod
    def raw_files(leaf_dir: Path) -> list[Path]:
        return sorted(list(leaf_dir.glob("*.tsv")) + list(leaf_dir.glob("*.csv")))

    def _read_and_format_file(
        self, file: Path, id_col: str
    ) -> tuple[pd.DataFrame | None, str | None]:
        parts = file.stem.split("_")
        sample_id = parts[0]
        sample_name = "_".join(parts[1:]) if len(parts) > 1 else None
        sep = "\t" if file.suffix.lower() == ".tsv" else ";"
        try:
            df = pd.read_csv(file, sep=sep)
        except Exception:
            df = pd.read_csv(file, sep=None, engine="python")
        if id_col not in df.columns:
            print(f"[ERROR] Warning: Column '{id_col}' not found in {file.name}. Skipping file.")
            return None, None
        df[id_col] = sample_id + "_" + df[id_col].astype(str)
        if id_col != "Accession":
            df = df.rename(columns={id_col: "Accession"})
        return df, sample_name

    def _save_merged_df(
        self,
        dfs: list[pd.DataFrame],
        rel_dir: Path,
        sample_name: str | None,
    ) -> None:
        merged = pd.concat(dfs, ignore_index=True)
        out_dir = self.raw_merged_dir / rel_dir.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{sample_name or rel_dir.name}_M.csv"
        output_path = out_dir / output_filename
        merged.to_csv(output_path, sep=";", index=False)
        self.written.append(output_path)
        print(f"[INFO] Merged: {rel_dir}")
        print(f"       Saved to: {output_path} (Shape: {merged.shape})")
