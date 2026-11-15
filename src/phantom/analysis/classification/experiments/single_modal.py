"""
One single-modal classification run end to end: load a feature matrix,
filter it, validate a random forest / XGBoost / CatBoost classifier by
leave-one-out and/or repeated stratified CV, and save the metrics, feature
importances, predictions and ROC curves.
"""

import os
from datetime import datetime

from phantom.analysis.classification.data.labeling import Labeling
from phantom.analysis.classification.data.loading import FeatureMatrixLoader
from phantom.analysis.classification.data.variance_filter import NearZeroVarianceFilter
from phantom.analysis.classification.machine_learning.cross_validation import (
    LeaveOneOutCrossValidator,
    RepeatedCrossValidator,
)
from phantom.analysis.classification.machine_learning.metrics import SupervisedEvaluator
from phantom.analysis.classification.machine_learning.models import MODELS
from phantom.analysis.classification.reporting.logger import Logger
from phantom.analysis.classification.reporting.result_files import ResultFileWriter
from phantom.analysis.classification.reporting.run_plots import ResultPlots
from phantom.analysis.classification.reporting.text_reports import ReportFormatter


class SingleModalClassifier:
    def __init__(self, config):
        self.config = config
        self.logger = None
        self.saver = None
        self.values = None
        self.labels = None
        self.feature_names = None
        self.sample_ids = None
        self.model = None

    def run(self):
        self._setup()
        self._load()
        self._build_model()
        self._validate()

    def _setup(self):
        _exp_dir, self.logger, self.saver = self._create_experiment()
        self.logger.info("=== SINGLE-MODAL CLASSIFIER ===")
        self.saver.save_metadata({**vars(self.config), "labels": Labeling.default().provenance()})

    def _create_experiment(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.config.vtool is not None and self.config.modality is not None:
            exp_name = f"run_{self.config.model_type}_{self.config.vtool}_{self.config.modality}_{timestamp}"
        else:
            exp_name = f"run_{self.config.model_type}_{timestamp}"
        exp_dir = os.path.join(self.config.out_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        logger = Logger.setup_logger(log_dir=exp_dir, log_filename=f"{timestamp}.log")
        saver = ResultFileWriter(exp_dir=exp_dir)
        return exp_dir, logger, saver

    def _load(self):
        loader = FeatureMatrixLoader(input_path=self.config.in_file, logger=self.logger)
        X, self.labels, self.sample_ids = loader.load()
        self.logger.info("Applying Near-Zero Variance Filter...")
        variance_filter = NearZeroVarianceFilter(logger=self.logger, threshold=4e-5)
        self.values, self.feature_names = variance_filter.fit_transform(X)

    def _build_model(self):
        self.model = MODELS[self.config.model_type](
            use_smote=self.config.use_smote, seed=self.config.seed
        )

    def _validate(self):
        if self.config.run_loocv:
            self._run_validation("loocv", LeaveOneOutCrossValidator(verbose=True))
        if self.config.run_repeated:
            validator = RepeatedCrossValidator(random_state=self.config.seed, verbose=True)
            self._run_validation("rcv", validator)

    def _run_validation(self, name, validator):
        self.logger.info(f"--- SINGLE-MODAL EVALUATION: {name.upper()} ---")
        results = validator.run(self.model, self.values, self.labels)
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
        self.logger.info(
            ReportFormatter.format_top_features(
                results.importance_mean, self.feature_names, self.values, self.labels
            )
        )

    def _save_results(self, name, results, metrics):
        mapped_ids = (
            self.sample_ids.iloc[results.test_indices].tolist()
            if results.test_indices is not None
            else None
        )
        self.saver.save_feature_importance(
            results.importance_mean, self.feature_names, self.values, self.labels, subfolder=name
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
            title=f"ROC Curve -- {self.config.model_type.upper()} ({name.upper()})",
            save_path=os.path.join(self.saver.exp_dir, name, "roc_curve.pdf"),
        )

    def _finish(self):
        self.logger.info("=== END EXPERIMENT ===")
        Logger.close(self.logger)
