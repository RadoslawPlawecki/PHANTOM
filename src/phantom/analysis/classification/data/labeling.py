"""
Single source of truth for the patient -> class-label mapping used across
the classification stage, nested-CV / permutation evaluation and feature
analytics.

Labels are *not* derived from sample ids in code. They are read from the
``label`` column of the merged metadata CSV (config.toml's
``[genomes].metadata_path``), written by ``preprocessing.samples`` from the
class folders of the raw folder (or by ``preprocessing.metadata``'s labeling strategies,
``analysis/genomes/metadata/label_strategies.py``).
"""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from phantom.analysis.genomes.metadata.table import MetadataTable
from phantom.config.loader import ConfigLoader

LABEL_COLUMN = "label"
VALID_LABELS = {0, 1}


class Labeling:
    def __init__(self, labels: pd.Series, source: Path | None = None):
        """
        ``labels`` maps a sample id (the metadata CSV's ``id``, e.g. "S12")
        to its integer class label.
        """
        labels = pd.Series(labels)
        invalid = set(labels.unique()) - VALID_LABELS
        if invalid:
            raise ValueError(
                f"Labels must be one of {sorted(VALID_LABELS)}, got {sorted(invalid)}."
            )
        if labels.index.has_duplicates:
            dupes = labels.index[labels.index.duplicated()].unique().tolist()
            raise ValueError(f"Duplicate sample ids in labels: {dupes}")
        self.labels = labels.astype(int)
        self.source = Path(source) if source is not None else None

    @classmethod
    def from_metadata(cls, path: Path | str) -> Labeling:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Metadata file not found: {path}. Run preprocessing.samples first."
            )
        df = pd.read_csv(path, sep=";")
        if LABEL_COLUMN not in df.columns:
            raise ValueError(
                f"No '{LABEL_COLUMN}' column in {path}. Run preprocessing.samples (labels "
                'from the raw folder\'s class folders) or preprocessing.metadata with targets = ["labeling"].'
            )
        df = df[["id", LABEL_COLUMN]].dropna(subset=[LABEL_COLUMN])
        return cls(df.set_index("id")[LABEL_COLUMN], source=path)

    @classmethod
    def default(cls) -> Labeling:
        """
        Labels from the metadata CSV of the active configuration (the one
        ``phantom --config`` was started with, see ConfigLoader.activate). Cached per
        (path, modification time), so repeated calls (e.g. one FeatureMatrixLoader
        per Optuna trial) don't re-read the file, while relabeling in the
        same session is still picked up.
        """
        path = MetadataTable.path_from_config(ConfigLoader().load())
        mtime = path.stat().st_mtime_ns if path.is_file() else None
        return _cached_from_metadata(path, mtime)

    @staticmethod
    def sample_id(ids: pd.Series) -> pd.Series:
        """
        Reduces any id form used across the pipeline to the metadata
        ``id``: "S12" (metadata), "geN|S12" (patient id in feature
        matrices) and "geN|S12_k141_123" (contig Accession, which may carry
        further "|"-separated suffixes such as "|provirus_...") all -> "S12".
        """
        ids = pd.Series(ids, dtype="string")
        return ids.str.split("|", n=1).str[-1].str.split("_").str[0].str.split("|").str[0]

    def derive_label(self, ids: pd.Series) -> np.ndarray:
        sample_ids = self.sample_id(ids)
        mapped = sample_ids.map(self.labels)
        missing = sample_ids[mapped.isna()].unique().tolist()
        if missing:
            where = f" in {self.source}" if self.source else ""
            raise ValueError(f"No label{where} for sample id(s): {missing}")
        return mapped.to_numpy(dtype=int)

    def provenance(self) -> dict:
        """
        What a run's metadata.json records about the labels it used: where
        they came from, a content hash of the full id -> label set, and
        the class balance.
        """
        ordered = self.labels.sort_index()
        payload = "\n".join(f"{i};{v}" for i, v in ordered.items())
        counts = ordered.value_counts().sort_index()
        return {
            "source": str(self.source) if self.source else None,
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "n_samples": len(ordered),
            "class_counts": {int(k): int(v) for k, v in counts.items()},
        }


@cache
def _cached_from_metadata(path: Path, _mtime: int | None) -> Labeling:
    return Labeling.from_metadata(path)
