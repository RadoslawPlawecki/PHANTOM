"""
Metrics of a classification (ROC AUC, balanced accuracy, F1, precision,
recall, PR AUC, MCC, G-mean, misclassified samples) and of the unsupervised
analysis (silhouette, PERMANOVA) from a distance matrix.
"""

from typing import ClassVar

import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


class SupervisedEvaluator:
    METRICS: ClassVar[dict] = {
        "roc_auc": lambda y_true, y_pred, y_prob: roc_auc_score(y_true, y_prob),
        "bacc": lambda y_true, y_pred, y_prob: balanced_accuracy_score(y_true, y_pred),
        "f1": lambda y_true, y_pred, y_prob: f1_score(y_true, y_pred),
        "precision": lambda y_true, y_pred, y_prob: precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": lambda y_true, y_pred, y_prob: recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": lambda y_true, y_pred, y_prob: average_precision_score(y_true, y_prob),
        "mcc": lambda y_true, y_pred, y_prob: matthews_corrcoef(y_true, y_pred),
        "gmean": lambda y_true, y_pred, y_prob: geometric_mean_score(y_true, y_pred),
    }

    @staticmethod
    def evaluate(y_true, y_pred, y_prob, test_indices):
        results = {}
        for name, metric_function in SupervisedEvaluator.METRICS.items():
            score = metric_function(y_true, y_pred, y_prob)
            metric_result = {"score": score}
            results[name] = metric_result
        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        results["specificity"] = {
            "score": true_negatives / (true_negatives + false_positives)
            if (true_negatives + false_positives) > 0
            else 0.0
        }
        results["sensitivity"] = {"score": results["recall"]["score"]}
        results["npv"] = {
            "score": true_negatives / (true_negatives + false_negatives)
            if (true_negatives + false_negatives) > 0
            else 0.0
        }
        results["confusion_matrix"] = {
            "TP": int(true_positives),
            "TN": int(true_negatives),
            "FP": int(false_positives),
            "FN": int(false_negatives),
        }
        if test_indices is not None:
            false_positive_positions = np.where((y_true == 0) & (y_pred == 1))[0]
            false_negative_positions = np.where((y_true == 1) & (y_pred == 0))[0]
            false_positive_indices = test_indices[false_positive_positions].tolist()
            false_negative_indices = test_indices[false_negative_positions].tolist()
            results["misclassified"] = {
                "fp": false_positive_indices,
                "fn": false_negative_indices,
            }
        else:
            results["misclassified"] = None
        return results


class UnsupervisedEvaluator:
    METRICS: ClassVar[dict] = {
        "silhouette": lambda dist_matrix, labels, seed: silhouette_score(
            dist_matrix, labels, metric="precomputed"
        ),
        "permanova": lambda dist_matrix, labels, seed: permanova(
            dist_matrix, labels, permutations=999, seed=seed
        ),
    }

    @staticmethod
    def evaluate(dist_matrix, labels, seed=42):
        results = {}
        for name, metric_function in UnsupervisedEvaluator.METRICS.items():
            result = metric_function(dist_matrix, labels, seed)
            if name == "silhouette":
                results[name] = {"score": float(result)}
            else:
                statistic, p_value = result
                results[name] = {"statistic": float(statistic), "p_value": float(p_value)}
        return results


def _pseudo_f(squared_distances: np.ndarray, groups: np.ndarray, n_groups: int) -> float:
    """PERMANOVA's pseudo-F: between-group over within-group sum of squared
    distances, each divided by its degrees of freedom."""
    n = len(groups)
    total = squared_distances.sum() / n / 2
    within = 0.0
    for group in range(n_groups):
        members = np.flatnonzero(groups == group)
        within += squared_distances[np.ix_(members, members)].sum() / 2 / len(members)
    return ((total - within) / (n_groups - 1)) / (within / (n - n_groups))


def permanova(distance_matrix, labels, permutations: int = 999, seed: int = 42):
    """
    PERMANOVA (Anderson 2001): do the groups differ more than chance would
    make them? Returns (pseudo-F, p-value). The p-value is the share of
    random relabelings (plus the real one) whose pseudo-F is at least the
    observed one.

    Same statistic and the same permutations as scikit-bio's ``permanova``
    (identical results), but a permuted pseudo-F that equals the observed one
    up to floating-point rounding counts as a tie: summing the same numbers in
    another order changes the last digit, which made scikit-bio's p-value flip
    between runs for distance matrices with many ties (e.g. Jaccard).
    """
    _, groups = np.unique(labels, return_inverse=True)
    n_groups = int(groups.max()) + 1
    squared = np.asarray(distance_matrix, dtype=np.float64) ** 2
    observed = _pseudo_f(squared, groups, n_groups)
    generator = np.random.default_rng(seed)
    permuted = np.array(
        [_pseudo_f(squared, generator.permutation(groups), n_groups) for _ in range(permutations)]
    )
    at_least_as_large = (permuted >= observed - 1e-9 * abs(observed)).sum()
    return observed, (at_least_as_large + 1) / (permutations + 1)
