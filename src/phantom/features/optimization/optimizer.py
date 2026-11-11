"""
Orchestrator for feature-engineering optimization -- ties the three
strategies (strategies/exploratory.py, nested_cv.py, permutation_search.py)
behind one questionary-driven entry point. Used by FeatureController's
"Optimize features" step and by running this module directly.
"""

import json
import os
from pathlib import Path

import pandas as pd

from phantom.classification.data.labeling import Labeling
from phantom.cli.features import FeatureOptimizationPrompts
from phantom.config.features import FeatureConfigManager

from .search_core import SEARCH_FNS, RunConfig, _load_preprocessed
from .strategies.exploratory import run_exploratory


class FeatureOptimizer:
    def __init__(self, config_mgr: FeatureConfigManager | None = None):
        self.config_mgr = config_mgr

    def run(self) -> None:
        """Legacy interactive menu: ask for everything, then optimize()."""
        tool = FeatureOptimizationPrompts.ask_tool_choice(list(SEARCH_FNS))
        if not tool:
            return
        mode = FeatureOptimizationPrompts.ask_mode_choice()
        if not mode:
            return

        config = RunConfig(
            model=FeatureOptimizationPrompts.ask_model_choice(),
            validator=FeatureOptimizationPrompts.ask_validator_choice(),
            metric=FeatureOptimizationPrompts.ask_metric_choice(),
            smote=FeatureOptimizationPrompts.ask_smote_choice(),
        )

        knobs: dict = {}
        if mode in ("nested", "all"):
            knobs["outer_folds"] = FeatureOptimizationPrompts.ask_outer_folds()
            knobs["outer_repeats"] = FeatureOptimizationPrompts.ask_outer_repeats()
            knobs["inner_trials"] = FeatureOptimizationPrompts.ask_inner_trials()
            knobs["top_k_features"] = FeatureOptimizationPrompts.ask_top_k_features()
        if mode in ("permutation", "all"):
            knobs["n_permutations"] = FeatureOptimizationPrompts.ask_n_permutations()
            knobs["permutation_trials"] = FeatureOptimizationPrompts.ask_permutation_trials()

        out_dir = FeatureOptimizationPrompts.ask_optimization_out_dir(tool)
        source_path, loaded_df = _load_preprocessed(tool, self.config_mgr)
        self.optimize(tool, mode, config, out_dir, source_path, loaded_df, **knobs)

    def optimize(
        self,
        tool: str,
        mode: str,
        config: RunConfig,
        out_dir,
        source_path: Path,
        loaded_df: pd.DataFrame,
        *,
        outer_folds: int | None = None,
        outer_repeats: int | None = None,
        inner_trials: int | None = None,
        top_k_features: int | None = None,
        n_permutations: int | None = None,
        permutation_trials: int | None = None,
    ) -> None:
        """
        Runs the evaluation strategies of ``mode`` on an already loaded and
        masked dataframe (see search_core.load_masked) and saves everything
        under ``out_dir``. Knobs a mode does not use must be None, so the
        recorded metadata only shows what actually applied.
        """
        os.makedirs(out_dir, exist_ok=True)

        from phantom.classification.analytics.saver import ExperimentSaver

        ExperimentSaver(exp_dir=str(out_dir)).save_metadata(
            {
                "tool": tool,
                "mode": mode,
                "model": config.model,
                "validator": config.validator,
                "metric": config.metric,
                "smote": config.smote,
                "outer_folds": outer_folds,
                "outer_repeats": outer_repeats,
                "inner_trials": inner_trials,
                "top_k_features": top_k_features,
                "n_permutations": n_permutations,
                "permutation_trials": permutation_trials,
                "labels": Labeling.default().provenance(),
            }
        )

        observed = None
        if mode in ("exploratory", "all"):
            study = run_exploratory(tool, loaded_df, config, out_dir, file_path=source_path)
            observed = study.best_value
            if mode == "exploratory":
                return

        if mode in ("permutation", "all"):
            if n_permutations is None:
                raise ValueError("n_permutations must be set when mode requires permutation")
            _run_mode_permutation(
                config,
                tool,
                loaded_df,
                out_dir,
                n_permutations=n_permutations,
                permutation_trials=permutation_trials,
                observed_score=observed,
            )
        if mode in ("nested", "all"):
            if outer_folds is None or outer_repeats is None or top_k_features is None:
                raise ValueError(
                    "outer_folds, outer_repeats, and top_k_features must be set "
                    "when mode requires nested CV"
                )
            _run_mode_nested(
                config,
                tool,
                loaded_df,
                out_dir,
                outer_folds=outer_folds,
                outer_repeats=outer_repeats,
                inner_trials=inner_trials,
                top_k_features=top_k_features,
            )


def _run_mode_nested(
    config: RunConfig,
    tool: str,
    df: pd.DataFrame,
    out_dir,
    outer_folds: int,
    outer_repeats: int,
    inner_trials: int | None,
    top_k_features: int,
) -> None:
    from phantom.classification.analytics.reporter import ReportFormatter
    from phantom.classification.analytics.saver import ExperimentSaver
    from phantom.classification.ml.evaluator import EvaluatorSl

    from .strategies.nested_cv import run_nested_cv

    result = run_nested_cv(
        tool,
        df,
        model_name=config.model,
        validator_name=config.validator,
        target_metric=config.metric,
        use_smote=config.smote,
        outer_folds=outer_folds,
        outer_repeats=outer_repeats,
        inner_trials=inner_trials,
        top_k=top_k_features,
    )
    full_metrics = EvaluatorSl.evaluate(
        result.full.y_true, result.full.y_pred, result.full.y_prob, result.full.patient_ids
    )
    ablation_metrics = EvaluatorSl.evaluate(
        result.ablation.y_true,
        result.ablation.y_pred,
        result.ablation.y_prob,
        result.ablation.patient_ids,
    )
    print(ReportFormatter.format_nested_comparison(full_metrics, ablation_metrics, config.metric))

    nested_dir = os.path.join(out_dir, "nested")
    os.makedirs(nested_dir, exist_ok=True)
    saver = ExperimentSaver(exp_dir=nested_dir)
    saver.save_metrics(dict(full_metrics), subfolder="full")
    saver.save_predictions(
        result.full.y_true,
        result.full.y_pred,
        result.full.y_prob,
        sample_ids=result.full.patient_ids,
        folds=result.full.fold,
        repeats=result.full.repeat,
        subfolder="full",
    )
    saver.save_metrics(dict(ablation_metrics), subfolder="ablation")
    saver.save_predictions(
        result.ablation.y_true,
        result.ablation.y_pred,
        result.ablation.y_prob,
        sample_ids=result.ablation.patient_ids,
        folds=result.ablation.fold,
        repeats=result.ablation.repeat,
        subfolder="ablation",
    )
    with open(os.path.join(nested_dir, "nested_meta.json"), "w") as f:
        json.dump(
            {
                "per_fold_best_params": result.per_fold_best_params,
                "per_fold_top_features": result.per_fold_top_features,
                "skipped_folds": result.skipped_folds,
            },
            f,
            indent=2,
            default=str,
        )


def _run_mode_permutation(
    config: RunConfig,
    tool: str,
    df: pd.DataFrame,
    out_dir,
    n_permutations: int,
    permutation_trials: int | None,
    observed_score: float | None = None,
) -> None:
    from phantom.classification.analytics.reporter import ReportFormatter

    from .strategies.permutation_search import run_permutation_test

    result = run_permutation_test(
        tool,
        df,
        model_name=config.model,
        validator_name=config.validator,
        target_metric=config.metric,
        use_smote=config.smote,
        n_permutations=n_permutations,
        permutation_trials=permutation_trials,
        observed_score=observed_score,
    )
    print(ReportFormatter.format_permutation_test(result, config.metric))

    perm_dir = os.path.join(out_dir, "permutation")
    os.makedirs(perm_dir, exist_ok=True)
    with open(os.path.join(perm_dir, "permutation_test.json"), "w") as f:
        json.dump(
            {
                "tool": result.tool,
                "observed_score": result.observed_score,
                "null_scores": result.null_scores.tolist(),
                "p_value": result.p_value,
                "baseline_score": result.baseline_score,
                "n_permutations": result.n_permutations,
                "permutation_trials": result.permutation_trials,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    FeatureOptimizer().run()
