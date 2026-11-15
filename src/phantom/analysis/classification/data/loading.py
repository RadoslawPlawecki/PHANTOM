"""
Finds and loads one feature matrix (``*_FEAT.csv``: an ``id`` column, one column per
feature) with the class label of every sample, taken from the metadata CSV.
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from phantom.analysis.classification.data.labeling import Labeling
from phantom.config.feature_versions import FeatureConfigManager


class EmptyFeatureMatrix(ValueError):
    """A feature matrix without a single sample."""


class FeatureMatrixLoader:
    def __init__(self, input_path: str, logger=None, labeling: Labeling | None = None):
        self.input_path = Path(input_path)
        self.logger = logger
        self.labeling = labeling
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

    def load(self):
        try:
            with open(self.input_path, newline="") as f:
                sample = f.read(4096)
                dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
                sep = dialect.delimiter
        except (csv.Error, OSError):
            sep = ";"
        df = pd.read_csv(self.input_path, sep=sep)
        if df.empty:
            raise EmptyFeatureMatrix(
                f"{self.input_path} has no samples (everything was filtered out before "
                "extraction, e.g. by the CheckV mask) -- nothing to classify"
            )
        sample_ids = df["id"].copy()
        y = (self.labeling or Labeling.default()).derive_label(sample_ids)
        X = df.drop(columns=["id", "label"], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce")
        self._log_load(X=X, y=y)
        return X, y, sample_ids

    def _log_load(self, X: pd.DataFrame, y: np.ndarray) -> None:
        counts = Counter(y)
        total = len(y)
        msg1 = f"Dataset Loaded: {self.input_path}"
        msg2 = f"Samples: {total}"
        msg3 = f"Features: {X.shape[1]}"
        class_distribution = "Class Distribution:"
        class_lines = []
        for cls, count in sorted(counts.items()):
            percent = 100 * count / total
            class_lines.append(f"       class {cls}: {count} ({percent:.1f}%)")
        joined_class_lines = "\n".join(class_lines)
        msg4 = f"\n{class_distribution}\n{joined_class_lines}\n"
        msg5 = f"Missing Values: {int(X.isna().sum().sum())}"
        if self.logger:
            self.logger.info(msg1)
            self.logger.info(msg2)
            self.logger.info(msg3)
            self.logger.info(msg4)
            self.logger.info(msg5)
        else:
            print(msg1)
            print(msg2)
            print(msg3)
            print(msg4)
            print(msg5)


# finding the feature file of a modality


def feature_files(
    feature_version_manager: FeatureConfigManager, version: str, modality_config: dict
) -> tuple[Path, list[Path]]:
    """
    All extracted ``*_<tag>_FEAT.csv`` files of one modality (a
    ``[modalities.<key>]`` table: ``tool``, ``tag``) in a feature version,
    plus the directory they were searched in.
    """
    extracted_dir = (
        feature_version_manager.get_stage_dir(version, "extracted") / modality_config["tool"]
    )
    return extracted_dir, sorted(extracted_dir.glob(f"*_{modality_config['tag']}_FEAT.csv"))


def find_feature_file(
    feature_version_manager: FeatureConfigManager,
    version: str,
    modality_config: dict,
    virus_identification_tool: str,
) -> Path:
    """
    The extracted feature file of one modality for one virus identification
    tool (``<vtool>_..._<tag>_FEAT.csv``). Raises FileNotFoundError naming the
    searched directory if there is none.
    """
    extracted_dir, files = feature_files(feature_version_manager, version, modality_config)
    match = next((f for f in files if f.name.startswith(f"{virus_identification_tool}_")), None)
    if match is None:
        raise FileNotFoundError(
            f"No {virus_identification_tool} {modality_config['tag']} feature file (*_{modality_config['tag']}_FEAT.csv) "
            f"in {extracted_dir}"
        )
    return match
