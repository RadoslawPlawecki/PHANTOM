"""Models, cross-validation, metrics and the data steps in front of them, on
small synthetic matrices with a known signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phantom.analysis.classification.data.alignment import ModalityAligner
from phantom.analysis.classification.data.labeling import Labeling
from phantom.analysis.classification.data.loading import EmptyFeatureMatrix, FeatureMatrixLoader
from phantom.analysis.classification.data.variance_filter import NearZeroVarianceFilter
from phantom.analysis.classification.machine_learning.cross_validation import (
    EarlyFusionLeaveOneOutCrossValidator,
    EarlyFusionRepeatedCrossValidator,
    LateFusionLeaveOneOutCrossValidator,
    LateFusionRepeatedCrossValidator,
    LeaveOneOutCrossValidator,
    RepeatedCrossValidator,
)
from phantom.analysis.classification.machine_learning.metrics import (
    SupervisedEvaluator,
    UnsupervisedEvaluator,
)
from phantom.analysis.classification.machine_learning.models import (
    MODELS,
    NO_BUILTIN_IMPORTANCE,
    MultiModalModel,
    get_logreg_model,
    get_rf_model,
)
from phantom.analysis.classification.machine_learning.significance import Significance


def _small_rf(seed=0):
    """Validator tests are about the splitting, not the forest: 10 trees are enough."""
    return get_rf_model(seed=seed, n_estimators=10)


def _data(n=24, informative=2, noise=3, seed=0):
    """Labels 0/1 half and half; the first ``informative`` columns separate them."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    signal = y[:, None] * 2.0 + rng.normal(0, 0.5, (n, informative))
    X = np.hstack([signal, rng.normal(0, 1, (n, noise))])
    return X, y


# models


@pytest.mark.parametrize("name", sorted(MODELS))
def test_each_model_learns_a_clear_signal(name):
    X, y = _data(n=80)  # XGBoost's min_child_weight = 8 needs some samples
    model = MODELS[name](seed=1).fit(X, y)
    assert (model.predict(X) == y).mean() >= 0.9
    assert model.predict_proba(X).shape == (len(y), 2)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_each_model_reports_importances_or_says_it_has_none(name):
    X, y = _data(n=80)
    importances = np.asarray(MODELS[name](seed=1).fit(X, y).feature_importances_)
    assert importances.shape == (X.shape[1],)
    if name in NO_BUILTIN_IMPORTANCE:
        assert np.isnan(importances).all()
    else:
        assert importances[:2].sum() > importances[2:].sum()  # the informative columns


@pytest.mark.parametrize("name", sorted(MODELS))
def test_each_model_is_reproducible_with_its_seed(name):
    X, y = _data(n=40)
    a = MODELS[name](seed=3, use_smote=True).fit(X, y).predict_proba(X)
    b = MODELS[name](seed=3, use_smote=True).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_each_model_runs_through_both_validators(name):
    X, y = _data(n=12)
    loocv = LeaveOneOutCrossValidator(verbose=False).run(MODELS[name](seed=0), X, y)
    assert sorted(loocv.test_indices) == list(range(12))
    rcv = RepeatedCrossValidator(n_splits=3, n_repeats=1, random_state=0, verbose=False).run(
        MODELS[name](seed=0), X, y
    )
    assert len(rcv.y_true) == 12
    assert not np.isnan(rcv.importance_mean).any()  # permutation importances: every model


def test_the_stages_offer_exactly_the_registered_models():
    from typing import get_args

    from phantom.stages._common import Model

    assert set(get_args(Model)) == set(MODELS)


def test_logistic_regression_standardizes_its_features():
    X, y = _data(n=40)
    model = get_logreg_model(seed=0).fit(X * 1000, y)
    assert [name for name, _ in model.model_.steps] == ["standardscaler", "logisticregression"]


def test_the_seed_reaches_the_estimator_and_smote():
    model = get_rf_model(use_smote=True, seed=7)
    assert model.model.random_state == 7 and model.random_state == 7


def test_smote_oversamples_only_inside_fit():
    X, y = _data(n=20)
    X, y = X[2:], y[2:]  # 9 vs 9 -> make it imbalanced
    keep = np.r_[np.where(y == 0)[0], np.where(y == 1)[0][:4]]
    model = get_rf_model(use_smote=True, seed=0).fit(X[keep], y[keep])
    assert model.predict(X[keep]).shape == (len(keep),)  # the data itself is untouched


def test_an_unfitted_model_has_no_importances():
    assert get_rf_model().feature_importances_.tolist() == [0.0]


def test_late_fusion_averages_the_modality_probabilities_with_weights():
    X, y = _data()
    blocks = {"a": X[:, :2], "b": X[:, 2:]}
    model = MultiModalModel(
        {"a": get_rf_model(seed=0), "b": get_rf_model(seed=0)}, weights={"a": 1.0, "b": 0.0}
    ).fit(blocks, y)
    only_a = get_rf_model(seed=0).fit(blocks["a"], y).predict_proba(blocks["a"])[:, 1]
    assert model.predict_proba(blocks)[:, 1] == pytest.approx(only_a)
    assert set(model.predict(blocks)) <= {0, 1}


# validators


def test_loocv_predicts_every_sample_once():
    X, y = _data(n=12)
    result = LeaveOneOutCrossValidator(verbose=False).run(_small_rf(0), X, y)
    assert sorted(result.test_indices) == list(range(12))
    assert result.importance_mean.shape == (X.shape[1],)


def test_repeated_cv_is_reproducible_with_a_seed_and_differs_without_it():
    X, y = _data(n=20)
    run = lambda seed: RepeatedCrossValidator(random_state=seed, verbose=False).run(  # noqa: E731
        _small_rf(seed), X, y
    )
    a, b, c = run(3), run(3), run(4)
    assert np.array_equal(a.y_prob, b.y_prob)
    assert np.array_equal(a.importance_mean, b.importance_mean)
    assert not np.array_equal(a.test_indices, c.test_indices)
    assert len(a.y_true) == 5 * len(y)  # 5 repeats x every sample once
    assert set(a.folds) == set(range(5)) and set(a.repeats) == set(range(5))


@pytest.mark.parametrize(
    "validator",
    [
        EarlyFusionLeaveOneOutCrossValidator(verbose=False),
        EarlyFusionRepeatedCrossValidator(n_splits=3, n_repeats=1, verbose=False),
    ],
)
def test_early_fusion_validators_split_importances_per_modality(validator):
    X, y = _data(n=12)
    data = {"a": {"values": X[:, :2]}, "b": {"values": X[:, 2:]}}
    result = validator.run(_small_rf(0), data, y)
    assert result.importance_mean["a"].shape == (2,)
    assert result.importance_mean["b"].shape == (3,)


@pytest.mark.parametrize(
    "validator",
    [
        LateFusionLeaveOneOutCrossValidator(verbose=False),
        LateFusionRepeatedCrossValidator(n_splits=3, n_repeats=1, verbose=False),
    ],
)
def test_late_fusion_validators_report_one_importance_vector_per_modality(validator):
    X, y = _data(n=12)
    data = {"a": {"values": pd.DataFrame(X[:, :2])}, "b": {"values": pd.DataFrame(X[:, 2:])}}
    model = MultiModalModel({"a": _small_rf(0), "b": _small_rf(0)})
    result = validator.run(model, data, y)
    assert sorted(result.importance_mean) == ["a", "b"]
    assert len(result.y_pred) == len(result.y_true)


# metrics


def test_supervised_metrics_of_a_known_prediction():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.2, 0.1])
    m = SupervisedEvaluator.evaluate(y_true, y_pred, y_prob, test_indices=np.arange(4))
    assert m["precision"]["score"] == 1.0
    assert m["recall"]["score"] == 0.5
    assert m["roc_auc"]["score"] == 1.0
    assert m["confusion_matrix"] == {"TP": 1, "TN": 2, "FP": 0, "FN": 1}
    assert m["misclassified"] == {"fp": [], "fn": [1]}


def test_unsupervised_metrics_are_reproducible_with_a_seed():
    rng = np.random.default_rng(0)
    points = np.r_[rng.normal(0, 1, (8, 2)), rng.normal(4, 1, (8, 2))]
    dist = np.linalg.norm(points[:, None] - points[None], axis=-1)
    labels = np.array([0] * 8 + [1] * 8)
    a = UnsupervisedEvaluator.evaluate(dist, labels, seed=5)
    b = UnsupervisedEvaluator.evaluate(dist, labels, seed=5)
    assert a == b
    assert a["silhouette"]["score"] > 0.5 and a["permanova"]["p_value"] < 0.05


def test_significance_helpers():
    assert Significance.permutation_pvalue(0.9, np.array([0.1, 0.2, 0.95])) == pytest.approx(0.5)
    assert Significance.majority_class_baseline(np.array([0, 0, 1]), "mcc") == 0.0


# data steps


def test_near_zero_variance_features_are_dropped():
    X = pd.DataFrame(
        {"constant": [1.0] * 6, "varies": [0, 1, 0, 1, 0, 1], "tiny": [0, 0, 0, 0, 0, 1e-9]}
    )
    values, names = NearZeroVarianceFilter(threshold=1e-6).fit_transform(X)
    assert names == ["varies"]
    assert np.asarray(values).shape == (6, 1)


def test_modalities_are_aligned_on_their_common_samples():
    raw = {
        "a": {"values": np.array([[1], [2], [3]]), "ids": ["S1", "S2", "S3"], "labels": [1, 0, 1]},
        "b": {"values": np.array([[9], [7]]), "ids": ["S3", "S1"], "labels": [1, 1]},
    }
    aligned, y, common = ModalityAligner.align(raw)
    assert common == ["S1", "S3"]
    assert aligned["a"]["values"].ravel().tolist() == [1, 3]
    assert aligned["b"]["values"].ravel().tolist() == [7, 9]
    assert y.tolist() == [1, 1]


def test_aligning_conflicting_labels_is_an_error():
    raw = {
        "a": {"values": np.array([[1]]), "ids": ["S1"], "labels": [1]},
        "b": {"values": np.array([[1]]), "ids": ["S1"], "labels": [0]},
    }
    with pytest.raises(ValueError, match="not consistent"):
        ModalityAligner.align(raw)


@pytest.mark.parametrize("sep", [";", ",", "\t"])
def test_a_feature_matrix_loads_with_its_labels_whatever_the_separator(tmp_path, sep):
    path = tmp_path / "m.csv"
    pd.DataFrame({"id": ["geN|S1", "geN|S2"], "f1": [1, 0], "f2": [0.5, 0.25]}).to_csv(
        path, sep=sep, index=False
    )
    labeling = Labeling(pd.Series({"S1": 1, "S2": 0}))
    X, y, ids = FeatureMatrixLoader(str(path), labeling=labeling).load()
    assert list(X.columns) == ["f1", "f2"]
    assert y.tolist() == [1, 0] and ids.tolist() == ["geN|S1", "geN|S2"]


def test_an_empty_feature_matrix_is_a_clear_error(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("id;f1\n", encoding="utf-8")
    with pytest.raises(EmptyFeatureMatrix, match="no samples"):
        FeatureMatrixLoader(str(path), labeling=Labeling(pd.Series({"S1": 1}))).load()


def test_a_missing_feature_matrix_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        FeatureMatrixLoader(str(tmp_path / "missing.csv"))


def test_permanova_matches_the_reference_implementation_and_ties_are_stable():
    from phantom.analysis.classification.machine_learning.metrics import permanova

    rng = np.random.default_rng(3)
    points = rng.normal(size=(20, 4)) + np.r_[np.zeros((10, 4)), np.ones((10, 4))]
    dist = np.linalg.norm(points[:, None] - points[None], axis=-1)
    labels = np.array([0] * 10 + [1] * 10)
    statistic, p_value = permanova(dist, labels, permutations=999, seed=5)
    # scikit-bio 0.7.4's permanova(DistanceMatrix(dist), labels, 999, seed=5) on the same data
    assert statistic == pytest.approx(3.685876844233047, rel=1e-12)
    assert p_value == 0.01
    # Binary features -> Jaccard distances with many ties: always the same p-value.
    binary = rng.integers(0, 2, (12, 6)).astype(bool)
    jaccard = np.array([[np.mean(a != b) if (a | b).any() else 0 for b in binary] for a in binary])
    six_and_six = np.array([0] * 6 + [1] * 6)
    results = {permanova(jaccard, six_and_six, seed=1) for _ in range(50)}
    assert len(results) == 1
