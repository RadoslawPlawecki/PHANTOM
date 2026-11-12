"""
Labeling strategies of ``preprocessing.metadata`` -- alternatives to the
labels of the sample sheet: a threshold on the sample number, a manual list
of cases, a regex on the sample id, or an external id,label CSV.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class LabelingStrategy(ABC):
    @abstractmethod
    def assign_labels(self, df: pd.DataFrame) -> dict[str, int]:
        pass


class ManualLabeler(LabelingStrategy):
    def __init__(self, positive_files: list[str]):
        self.positive_files = set(positive_files)

    def assign_labels(self, df: pd.DataFrame) -> dict[str, int]:
        return {
            row["id"]: (1 if row["id"] in self.positive_files else 0) for _, row in df.iterrows()
        }


class PatternLabeler(LabelingStrategy):
    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)

    def assign_labels(self, df: pd.DataFrame) -> dict[str, int]:
        return {
            row["id"]: (1 if self.pattern.search(str(row["id"])) else 0) for _, row in df.iterrows()
        }


class SampleThresholdLabeler(LabelingStrategy):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def assign_labels(self, df: pd.DataFrame) -> dict[str, int]:
        labels = {}
        for _, row in df.iterrows():
            match = re.search(r"\d+", str(row["id"]))
            sample_num = int(match.group()) if match else float("inf")
            labels[row["id"]] = 1 if sample_num <= self.threshold else 0
        return labels


class ExternalCSVLabeler(LabelingStrategy):
    # Joins on the metadata's sample ``id`` (e.g. "S12") by default: every
    # metadata row has it, whereas original file names are per-tool columns
    # (``geN_original_file``, ``VIB_original_file``, ...).
    def __init__(self, csv_path: Path, join_key: str = "id", label_col: str = "label"):
        self.csv_path = Path(csv_path)
        self.join_key = join_key
        self.label_col = label_col

    def assign_labels(self, df: pd.DataFrame) -> dict[str, int]:
        """
        Every sample must get an explicit label from the external CSV -- a
        sample missing from it is an error, never silently class 0.
        """
        # Sniff the delimiter: the project's own CSVs use ";", hand-made
        # manifests usually ",".
        external_df = pd.read_csv(self.csv_path, sep=None, engine="python")
        for name, frame in (("metadata", df), (str(self.csv_path), external_df)):
            if self.join_key not in frame.columns:
                raise ValueError(f"Join column '{self.join_key}' not found in {name}.")
        if self.label_col not in external_df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in {self.csv_path}.")
        dupes = external_df[self.join_key][external_df[self.join_key].duplicated()]
        if not dupes.empty:
            raise ValueError(
                f"Duplicate '{self.join_key}' values in {self.csv_path}: {dupes.unique().tolist()}"
            )
        merged = df.merge(
            external_df[[self.join_key, self.label_col]], on=self.join_key, how="left"
        )
        missing = merged.loc[merged[self.label_col].isna(), "id"].tolist()
        if missing:
            raise ValueError(f"No label in {self.csv_path} for sample id(s): {missing}")
        labels = merged[self.label_col]
        invalid = set(labels.unique()) - {0, 1}
        if invalid:
            raise ValueError(f"Labels must be 0 or 1, got {sorted(invalid)}.")
        return dict(zip(merged["id"], labels.astype(int), strict=True))
