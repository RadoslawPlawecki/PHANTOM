"""
Plain-text summaries for the experiment log: metrics, nested-CV and
permutation-test results, confusion matrices, the top features and the
misclassified samples.
"""

from collections import Counter

import numpy as np
import pandas as pd


class ReportFormatter:
    @staticmethod
    def format_metric(metric: dict) -> str:
        return f"{metric['score']:.3f}"

    @staticmethod
    def format_metrics(metrics: dict) -> str:
        return (
            "\nModel Performance:\n"
            f"ROC-AUC: {ReportFormatter.format_metric(metrics['roc_auc'])}\n"
            f"Balanced Accuracy: {ReportFormatter.format_metric(metrics['bacc'])}\n"
            f"F1 Score: {ReportFormatter.format_metric(metrics['f1'])}\n"
            f"Precision: {ReportFormatter.format_metric(metrics['precision'])}\n"
            f"Recall (Sensitivity): {ReportFormatter.format_metric(metrics['recall'])}\n"
            f"Specificity: {ReportFormatter.format_metric(metrics['specificity'])}\n"
            f"G-Mean: {ReportFormatter.format_metric(metrics['gmean'])}\n"
            f"NPV: {ReportFormatter.format_metric(metrics['npv'])}\n"
            f"Average Precision Score: {ReportFormatter.format_metric(metrics['pr_auc'])}\n"
            f"Matthews Correlation Coefficient: {ReportFormatter.format_metric(metrics['mcc'])}\n"
        )

    @staticmethod
    def format_nested_comparison(
        full_metrics: dict, ablation_metrics: dict, target_metric: str
    ) -> str:
        full_score = ReportFormatter.format_metric(full_metrics[target_metric])
        ablation_score = ReportFormatter.format_metric(ablation_metrics[target_metric])
        return (
            "\nNested CV -- Real-World Estimate vs. Feature-Importance Mini-Model:\n"
            f"Full feature matrix ({target_metric.upper()}): {full_score}\n"
            f"Top-K mini-model      ({target_metric.upper()}): {ablation_score}\n"
            "A mini-model score close to the full model's suggests the signal is "
            "carried by a small, coherent feature set rather than noise scattered "
            "across a wide sparse matrix.\n"
        )

    @staticmethod
    def format_permutation_test(result, target_metric: str) -> str:
        null_scores = result.null_scores
        return (
            "\nPermutation Test (does the relationship beat chance?):\n"
            f"Observed best score ({target_metric.upper()}): {result.observed_score:.4f}\n"
            f"Null distribution ({result.n_permutations} reps, "
            f"{result.permutation_trials} trials/rep): "
            f"mean={null_scores.mean():.4f}, std={null_scores.std():.4f}, "
            f"max={null_scores.max():.4f}\n"
            f"Majority-class baseline ({target_metric.upper()}): {result.baseline_score:.4f}\n"
            f"p-value: {result.p_value:.4f}\n"
        )

    @staticmethod
    def format_confusion_matrix(confusion: dict) -> str:
        return (
            "\nConfusion Matrix:\n"
            f"TP: {confusion['TP']:02d} | FP: {confusion['FP']:02d}\n"
            f"FN: {confusion['FN']:02d} | TN: {confusion['TN']:02d}\n"
        )

    @staticmethod
    def format_top_features(
        importances: np.ndarray, feature_names: list, X: np.ndarray, y: np.ndarray, k: int = 10
    ) -> str:
        lines = [
            "\nTop Features & Class Characteristics:",
            f"{'No.':>3} {'Feature':<33} | {'Importance':>10} | {'Mean Class 0':>12} | {'Mean Class 1':>12}",
            "-" * 100,
        ]
        importances = np.asarray(importances, dtype=float)
        if np.isnan(importances).all():
            return (
                "\nTop Features: not available -- the model has no built-in feature "
                "importances (the rcv validator reports permutation importances).\n"
            )
        indices = np.nan_to_num(importances, nan=-np.inf).argsort()[-k:][::-1]
        for i, j in enumerate(indices, 1):
            feat_values = X[:, j]
            mean_0 = np.mean(feat_values[y == 0])
            mean_1 = np.mean(feat_values[y == 1])
            lines.append(
                f"{i:>3}. "
                f"{feature_names[j]:<33} | "
                f"{importances[j]:>10.4f} | "
                f"{mean_0:>12.4f} | "
                f"{mean_1:>12.4f}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def format_misclassified_samples(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        test_indices: np.ndarray,
        sample_ids: pd.Series,
        top_k: int = 10,
    ) -> str:
        false_positive_indices = test_indices[(y_true == 0) & (y_pred == 1)]
        false_negative_indices = test_indices[(y_true == 1) & (y_pred == 0)]
        false_positive_counts = Counter(false_positive_indices)
        false_negative_counts = Counter(false_negative_indices)

        def format_block(title: str, counts: Counter) -> str:
            if not counts:
                return f"{title}: None"
            parts = []
            for indices, count in counts.most_common():
                label = str(sample_ids.iloc[indices])
                if count > 1:
                    parts.append(f"{label} [{count}]")
                else:
                    parts.append(label)
            return f"{title}: " + ", ".join(parts)

        false_positive_line = format_block("False Positives", false_positive_counts)
        false_negative_line = format_block("False Negatives", false_negative_counts)
        return (
            "\nMisclassified Samples:\n" + false_positive_line + "\n" + false_negative_line + "\n"
        )

    @staticmethod
    def format_unsupervised_metrics(results):
        lines = [
            "",
            "Unsupervised Evaluation",
            "-" * 80,
            f"{'Modality':<10} {'Silhouette':>12} {'PERMANOVA F':>14} {'PERMANOVA p':>14}",
            "-" * 80,
        ]
        for modality, metrics in results.items():
            lines.append(
                f"{modality.upper():<10}"
                f"{metrics['silhouette']['score']:>12.3f}"
                f"{metrics['permanova']['statistic']:>14.3f}"
                f"{metrics['permanova']['p_value']:>14.3f}"
            )
        lines.append("-" * 80)
        return "\n".join(lines)
