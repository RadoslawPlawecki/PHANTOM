"""
One multimodal classification run end-to-end: align the comp, host and
func feature matrices, fuse them (early: concatenated features; late:
weighted average of per-modality classifiers, weights optionally optimized
with Optuna), validate, and save the results.
"""

import os
from datetime import datetime

from phantom.analysis.classification.data.alignment import ModalityAligner
from phantom.analysis.classification.data.labeling import Labeling
from phantom.analysis.classification.data.loading import FeatureMatrixLoader
from phantom.analysis.classification.data.variance_filter import NearZeroVarianceFilter
from phantom.analysis.classification.machine_learning.cross_validation import (
    EarlyFusionLeaveOneOutCrossValidator,
    EarlyFusionRepeatedCrossValidator,
    LateFusionLeaveOneOutCrossValidator,
    LateFusionRepeatedCrossValidator,
)
from phantom.analysis.classification.machine_learning.fusion_weights import (
    LateFusionWeightOptimizer,
)
from phantom.analysis.classification.machine_learning.metrics import SupervisedEvaluator
from phantom.analysis.classification.machine_learning.models import (
    MODELS,
    MultiModalModel,
)
from phantom.analysis.classification.reporting.logger import Logger
from phantom.analysis.classification.reporting.result_files import ResultFileWriter
from phantom.analysis.classification.reporting.run_plots import ResultPlots
from phantom.analysis.classification.reporting.text_reports import ReportFormatter


class MultiModalClassifier:
    def __init__(self, config):
        self.config = config
        self.logger = None
        self.saver = None
        self.X_raw = {}
        self.X_data = None
        self.labels = None
        self.common_samples = None
        self.sample_ids = None
        self.feature_names_dict = {}
        self.model = None
        self.model_factories = MODELS

    def run(self):
        self._setup()
        self._load()
        self._build_model()
        self._validate()
        self._finish()

    def _setup(self):
        _exp_dir, self.logger, self.saver = self._create_experiment()
        fusion_name = {"early": "EARLY FUSION", "late": "LATE FUSION"}.get(self.config.fusion)
        if fusion_name is None:
            raise ValueError(f"Unsupported fusion strategy: {self.config.fusion}")
        self.logger.info(f"=== MULTIMODAL CLASSIFIER ({fusion_name}) ===")
        self.saver.save_metadata({**vars(self.config), "labels": Labeling.default().provenance()})

    def _create_experiment(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_config = self.config
        prefix = f"run_{run_config.model_type}_{run_config.vtool}_" if run_config.vtool else "run_"
        exp_name = f"{prefix}{run_config.fusion}_fusion_{timestamp}"
        exp_dir = os.path.join(self.config.out_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        logger = Logger.setup_logger(log_dir=exp_dir, log_filename=f"{timestamp}.log")
        saver = ResultFileWriter(exp_dir=exp_dir)
        return exp_dir, logger, saver

    def _load(self):
        paths = {
            "comp": self.config.comp,
            "func": self.config.func,
            "host": self.config.host,
        }
        for modality in paths:
            loader = FeatureMatrixLoader(input_path=paths[modality], logger=self.logger)
            X, labels, sample_ids = loader.load()
            variance_filter = NearZeroVarianceFilter(logger=self.logger, threshold=4e-5)
            values, feature_names = variance_filter.fit_transform(X)
            self.X_raw[modality] = {
                "values": values,
                "feature_names": feature_names,
                "ids": list(sample_ids),
                "labels": list(labels),
            }
        self.X_data, self.labels, self.common_samples = ModalityAligner.align(self.X_raw)
        self.feature_names_dict = {
            modality: self.X_data[modality]["feature_names"] for modality in self.X_data
        }
        self.sample_ids = sample_ids
        self.logger.info("Datasets Aligned")

    def _build_model(self):
        if self.config.fusion == "early":
            self.model = self.model_factories[self.config.model_type](
                use_smote=self.config.use_smote, seed=self.config.seed
            )
            self.loocv = EarlyFusionLeaveOneOutCrossValidator(verbose=True)
            self.rcv = EarlyFusionRepeatedCrossValidator(
                n_splits=5,
                n_repeats=5,
                random_state=self.config.seed,
                verbose=True,
            )
        elif self.config.fusion == "late":
            weights = self._get_weights()
            fusion_models = {
                modality: self.model_factories[self.config.model_type](
                    use_smote=self.config.use_smote, seed=self.config.seed
                )
                for modality in self.X_data
            }
            self.model = MultiModalModel(models_dict=fusion_models, weights=weights)
            self.loocv = LateFusionLeaveOneOutCrossValidator(verbose=True)
            self.rcv = LateFusionRepeatedCrossValidator(
                n_splits=5,
                n_repeats=5,
                random_state=self.config.seed,
                verbose=True,
            )
        else:
            raise ValueError(f"Unsupported fusion: {self.config.fusion}")

    def _get_weights(self):
        if not self.config.opt:
            self.logger.info("Using Default Late-Fusion Weights")
            return {
                "comp": 0.3132,
                "func": 0.8761,
                "host": 0.1119,
            }
        self.logger.info(f"Starting Optuna Optimization ({self.config.n_trials} Trials)...")
        return self._optimize()

    def _optimize(self):
        optimizer = LateFusionWeightOptimizer(
            self.model_factories,
            self.config,
            {
                "comp": self.config.comp,
                "func": self.config.func,
                "host": self.config.host,
            },
            self.logger,
            n_trials=self.config.n_trials,
            seed=self.config.seed,
        )
        return optimizer.optimize(self.X_data, self.labels)

    def _validate(self):
        if self.config.run_loocv:
            self._run_validation("loocv", self.loocv)
        if self.config.run_repeated:
            self._run_validation("rcv", self.rcv)

    def _run_validation(self, name, validator):
        self.logger.info(f"--- MULTIMODAL EVALUATION: {name.upper()} ---")
        results = validator.run(self.model, self.X_data, self.labels)
        metrics = SupervisedEvaluator.evaluate(
            results.y_true, results.y_pred, results.y_prob, results.test_indices
        )
        self._log_results(results, metrics)
        self._save_results(name, results, metrics)

    def _log_results(self, results, metrics):
        self.logger.info(ReportFormatter.format_metrics(metrics))
        self.logger.info(ReportFormatter.format_confusion_matrix(metrics["confusion_matrix"]))
        self.logger.info(
            ReportFormatter.format_misclassified_samples(
                results.y_true, results.y_pred, results.test_indices, self.sample_ids
            )
        )
        for modality, importances in results.importance_mean.items():
            self.logger.info(f"\n--- Feature Importance: {modality.upper()} ---")
            names = self.feature_names_dict[modality]
            self.logger.info(
                ReportFormatter.format_top_features(
                    importances, names, self.X_data[modality]["values"], self.labels
                )
            )

    def _save_results(self, name, results, metrics):
        mapped_ids = (
            self.sample_ids.iloc[results.test_indices].tolist()
            if results.test_indices is not None
            else None
        )
        for modality, importances in results.importance_mean.items():
            self.saver.save_feature_importance(
                importances=importances,
                feature_names=self.feature_names_dict[modality],
                X=self.X_data[modality]["values"],
                y=self.labels,
                subfolder=os.path.join(name, modality),
            )
        self.saver.save_predictions(
            y_true=results.y_true,
            y_pred=results.y_pred,
            y_prob=results.y_prob,
            sample_ids=mapped_ids,
            folds=getattr(results, "folds", None),
            repeats=getattr(results, "repeats", None),
            subfolder=name,
        )
        self.saver.save_metrics(metrics, subfolder=name)
        ResultPlots.plot_roc_curve(
            results.y_true,
            results.y_prob,
            title=f"ROC Curve -- {self.config.model_type.upper()} {self.config.fusion.upper()} FUSION ({name.upper()})",
            save_path=os.path.join(self.saver.exp_dir, name, "roc_curve.pdf"),
        )

    def _finish(self):
        self.logger.info("=== END EXPERIMENT ===")
        Logger.close(self.logger)
