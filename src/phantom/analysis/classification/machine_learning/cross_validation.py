"""
Cross-validation schemes -- leave-one-out and repeated stratified k-fold --
for single-modal data and for early and late fusion. Each returns the
out-of-fold predictions of every sample and the feature importances averaged
over the folds: the model's built-in importances for leave-one-out,
permutation importances (mean and standard deviation) for repeated k-fold.

Two things vary independently, so each is written once:

- the scheme (``_LeaveOneOut``, ``_Repeated``): how samples are split and
  how the folds' importances are summarized;
- the data (``_SingleModal``, ``_EarlyFusion``, ``_LateFusion``): how a fold's
  training and test data are cut out, how a model is fitted on them and
  where its importances come from.

Each validator below is one scheme applied to one kind of data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from tqdm import tqdm

PERMUTATION_REPEATS = 5

Importances = np.ndarray | dict[str, np.ndarray]
Parts = dict[Any, np.ndarray]


@dataclass
class CrossValidationResults:
    """Out-of-fold predictions (one entry per prediction, in fold order) and
    the importances averaged over the folds: one array, or one per modality."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    test_indices: np.ndarray
    importance_mean: Importances
    importance_std: Importances | None = None
    folds: np.ndarray | None = None  # repeated k-fold: fold and repeat of each prediction
    repeats: np.ndarray | None = None


class _SingleModal:
    """One feature matrix."""

    label = "Single-Modal"

    def prepare(self, X):
        return X

    def subset(self, X, indices):
        return X[indices]

    def fit(self, model, X, y):
        model.fit(X, y)  # the wrapper clones its estimator on every fit
        return model

    def builtin_importances(self, fitted) -> Parts:
        return {None: fitted.feature_importances_}

    def permutation_targets(self, fitted, X_test) -> dict[Any, tuple]:
        """{part: (estimator, its test data)} to permute features of."""
        return {None: (fitted.model_, X_test)}

    def finish(self, importances: Parts, X_data) -> Importances:
        return importances[None]


class _EarlyFusion(_SingleModal):
    """The modalities side by side as one matrix, one model; importances
    split back by modality."""

    label = "Multimodal Early Fusion"

    def prepare(self, X_data: dict):
        return np.hstack([np.asarray(block["values"]) for block in X_data.values()])

    def fit(self, model, X, y):
        return super().fit(clone(model), X, y)

    def permutation_targets(self, fitted, X_test):
        return {None: (getattr(fitted, "model_", fitted), X_test)}

    def finish(self, importances, X_data):
        flat, split, start = importances[None], {}, 0
        for modality, block in X_data.items():
            width = block["values"].shape[1]
            split[modality] = flat[start : start + width]
            start += width
        return split


class _LateFusion(_SingleModal):
    """One model per modality, their probabilities combined; importances per modality."""

    label = "Multimodal Late Fusion"

    def prepare(self, X_data: dict):
        return {modality: block["values"] for modality, block in X_data.items()}

    def subset(self, X: dict, indices):
        return {
            modality: values.iloc[indices] if hasattr(values, "iloc") else values[indices]
            for modality, values in X.items()
        }

    def fit(self, model, X, y):
        return super().fit(clone(model), X, y)

    def builtin_importances(self, fitted):
        return {
            modality: part.feature_importances_ for modality, part in fitted.models_dict.items()
        }

    def permutation_targets(self, fitted, X_test):
        return {
            modality: (fitted.models_dict[modality].model_, X_test[modality]) for modality in X_test
        }

    def finish(self, importances, X_data):
        return importances


class _CrossValidator(ABC):
    data: _SingleModal
    scheme: str

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(self, model, X_data, y: np.ndarray) -> CrossValidationResults:
        X = self.data.prepare(X_data)
        splits, total = self._splits(y)
        progress = tqdm(
            splits, total=total, desc=f"{self.data.label} {self.scheme}", disable=not self.verbose
        )
        y_true, y_pred, y_prob, tested, fold_of_each, fold_importances = [], [], [], [], [], []
        for fold, (train, test) in enumerate(progress):
            fitted = self.data.fit(model, self.data.subset(X, train), y[train])
            X_test = self.data.subset(X, test)
            y_true.extend(y[test])
            y_pred.extend(fitted.predict(X_test))
            y_prob.extend(fitted.predict_proba(X_test)[:, 1])
            tested.extend(test)
            fold_of_each.extend([fold] * len(test))
            fold_importances.append(self._importances(fitted, X_test, y[test]))
        return CrossValidationResults(
            y_true=np.array(y_true),
            y_pred=np.array(y_pred),
            y_prob=np.array(y_prob),
            test_indices=np.array(tested),
            **self._summary(fold_importances, np.array(fold_of_each), X_data, len(y)),
        )

    @abstractmethod
    def _splits(self, y) -> tuple[Iterator, int]:
        """(train, test) index pairs and how many there are."""

    @abstractmethod
    def _importances(self, fitted, X_test, y_test) -> Parts:
        """The importances of one fold's model."""

    @abstractmethod
    def _summary(self, fold_importances: list[Parts], fold_of_each, X_data, n_samples) -> dict:
        """The CrossValidationResults fields that summarize the folds."""


class _LeaveOneOut(_CrossValidator):
    """Every sample is the test set once; the models' built-in importances, averaged."""

    scheme = "LOOCV"

    def _splits(self, y):
        return LeaveOneOut().split(np.zeros((len(y), 1))), len(y)

    def _importances(self, fitted, X_test, y_test):
        return self.data.builtin_importances(fitted)

    def _summary(self, fold_importances, fold_of_each, X_data, n_samples):
        totals = {part: np.zeros(len(values)) for part, values in fold_importances[0].items()}
        for importances in fold_importances:
            for part, values in importances.items():
                totals[part] += values
        means = {part: total / n_samples for part, total in totals.items()}
        return {"importance_mean": self.data.finish(means, X_data)}


class _Repeated(_CrossValidator):
    """Stratified k-fold, repeated with reshuffling; permutation importances on
    each test fold, their mean and standard deviation over the folds."""

    scheme = "Repeated CV"

    def __init__(self, n_splits=5, n_repeats=5, random_state=42, verbose=True):
        super().__init__(verbose)
        self.n_splits = n_splits
        self.cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    def _splits(self, y):
        return self.cv.split(np.zeros((len(y), 1)), y), self.cv.get_n_splits()

    def _importances(self, fitted, X_test, y_test):
        targets = self.data.permutation_targets(fitted, X_test)
        return {
            part: permutation_importance(
                estimator,
                X_part,
                y_test,
                n_repeats=PERMUTATION_REPEATS,
                random_state=self.cv.random_state,
            ).importances_mean
            for part, (estimator, X_part) in targets.items()
        }

    def _summary(self, fold_importances, fold_of_each, X_data, n_samples):
        stacked = {
            part: np.vstack([importances[part] for importances in fold_importances])
            for part in fold_importances[0]
        }
        mean = {part: matrix.mean(axis=0) for part, matrix in stacked.items()}
        std = {part: matrix.std(axis=0) for part, matrix in stacked.items()}
        return {
            "importance_mean": self.data.finish(mean, X_data),
            "importance_std": self.data.finish(std, X_data),
            "folds": fold_of_each % self.n_splits,
            "repeats": fold_of_each // self.n_splits,
        }


class LeaveOneOutCrossValidator(_LeaveOneOut):
    data = _SingleModal()


class RepeatedCrossValidator(_Repeated):
    data = _SingleModal()


class EarlyFusionLeaveOneOutCrossValidator(_LeaveOneOut):
    data = _EarlyFusion()


class EarlyFusionRepeatedCrossValidator(_Repeated):
    data = _EarlyFusion()


class LateFusionLeaveOneOutCrossValidator(_LeaveOneOut):
    data = _LateFusion()


class LateFusionRepeatedCrossValidator(_Repeated):
    data = _LateFusion()
