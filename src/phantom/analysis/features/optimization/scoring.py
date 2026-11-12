"""
Scoring one feature-engineering configuration: the feature matrix it builds
is loaded with the samples' labels, near-zero-variance features are dropped,
and the cross-validated target metric of a classifier on it is the score the
Optuna search maximizes (0 when too few features or a single class remain).
"""

import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from phantom.analysis.classification.data.loading import FeatureMatrixLoader
from phantom.analysis.classification.data.variance_filter import NearZeroVarianceFilter
from phantom.analysis.classification.ml.evaluator import SupervisedEvaluator


class FeatureConfigScorer:
    def __init__(
        self,
        model: Any,
        validator: Any,
        target_metric: str = "mcc",
        nzv_threshold: float = 4e-5,
        min_features: int = 3,
        logger=None,
    ):
        self.model = model
        self.validator = validator
        self.target_metric = target_metric
        self.nzv_threshold = nzv_threshold
        self.min_features = min_features
        self.logger = logger

    def run(self, feature_matrix: pd.DataFrame, y_override: pd.Series | None = None) -> float:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            feature_matrix.to_csv(tmp.name, sep=";", index=False)
            tmp_path = tmp.name
        try:
            loader = FeatureMatrixLoader(input_path=tmp_path, logger=self.logger)
            X, labels, sample_ids = loader.load()
            if y_override is not None:
                mapped = sample_ids.map(y_override)
                if mapped.isna().any():
                    raise ValueError(
                        "y_override does not cover every patient id in the feature matrix."
                    )
                labels = mapped.to_numpy(dtype=int)
            if X.empty or X.shape[1] < self.min_features or len(np.unique(labels)) <= 1:
                return 0.0
            nzv = NearZeroVarianceFilter(logger=self.logger, threshold=self.nzv_threshold)
            values, _feature_names = nzv.fit_transform(X)
            if values.shape[1] == 0:
                return 0.0
            if isinstance(values, pd.DataFrame):
                values = values.copy()
            else:
                values = np.array(values, copy=True)
            results = self.validator.run(self.model, values, labels)
            metrics = SupervisedEvaluator.evaluate(
                results.y_true, results.y_pred, results.y_prob, results.test_idx
            )
            metric_value = metrics.get(self.target_metric, 0.0)
            if isinstance(metric_value, dict):
                metric_value = metric_value.get("score", 0.0)
            return float(metric_value)
        except KeyError:
            return 0.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
