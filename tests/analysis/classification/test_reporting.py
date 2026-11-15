"""What a classification run leaves behind: result files, text reports, the
log, the ROC/MDS plots, and the experiment figures built from all runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phantom.analysis.classification.reporting import experiment_figures as figures
from phantom.analysis.classification.reporting.logger import Logger
from phantom.analysis.classification.reporting.result_files import ResultFileWriter
from phantom.analysis.classification.reporting.run_plots import ResultPlots
from phantom.analysis.classification.reporting.text_reports import ReportFormatter

LABELS = figures.Labels(
    virus_identification_tools={"geN": "geNomad", "VIB": "VIBRANT"},
    modalities={"comp": "Virus taxonomy", "func": "Gene function"},
)


# result files and reports


def test_the_saver_writes_every_result_file(tmp_path):
    saver = ResultFileWriter(str(tmp_path))
    saver.save_metadata({"seed": np.int64(3), "weights": np.array([0.5, 0.5])})
    assert json.loads((tmp_path / "metadata.json").read_text()) == {
        "seed": 3,
        "weights": [0.5, 0.5],
    }
    metrics = {"mcc": {"score": 0.5}, "confusion_matrix": {"TP": 1}, "misclassified": {"fp": []}}
    saver.save_metrics(metrics, subfolder="loocv")
    assert json.loads((tmp_path / "loocv" / "metrics.json").read_text()) == {"mcc": {"score": 0.5}}
    assert json.loads((tmp_path / "loocv" / "confusion_matrix.json").read_text()) == {"TP": 1}
    X, y = np.array([[1.0, 0.0], [3.0, 2.0]]), np.array([0, 1])
    saver.save_feature_importance(np.array([0.1, 0.9]), ["a", "b"], X, y, subfolder="loocv")
    importance = pd.read_csv(tmp_path / "loocv" / "feature_importance.csv")
    assert importance["feature"].tolist() == ["b", "a"]  # most important first
    assert importance.set_index("feature")["mean_class_1"].to_dict() == {"b": 2.0, "a": 3.0}
    saver.save_predictions(
        [0, 1],
        [0, 0],
        [0.2, 0.4],
        sample_ids=["S1", "S2"],
        folds=[0, 1],
        repeats=[0, 0],
        subfolder="rcv",
    )
    predictions = pd.read_csv(tmp_path / "rcv" / "predictions.csv")
    assert list(predictions.columns) == [
        "y_true",
        "y_pred",
        "y_prob_0",
        "y_prob_1",
        "sample_id",
        "fold",
        "repeat",
    ]
    saver.save_unsupervised_metrics(
        {"comp": {"silhouette": {"score": 0.3}, "permanova": {"statistic": 2.0, "p_value": 0.01}}}
    )
    assert pd.read_csv(tmp_path / "unsupervised_metrics.csv").iloc[0].to_dict() == {
        "Modality": "Comp",
        "silhouette_score": 0.3,
        "permanova_F": 2.0,
        "permanova_p": 0.01,
    }


def test_text_reports():
    names = (
        "roc_auc",
        "bacc",
        "f1",
        "precision",
        "recall",
        "specificity",
        "gmean",
        "npv",
        "pr_auc",
    )
    metrics = {"mcc": {"score": 0.25}, **{n: {"score": 0.5} for n in names}}
    assert "Matthews Correlation Coefficient: 0.250" in ReportFormatter.format_metrics(metrics)
    assert "TP" in ReportFormatter.format_confusion_matrix({"TP": 1, "TN": 2, "FP": 0, "FN": 1})
    comparison = ReportFormatter.format_nested_comparison(metrics, metrics, "mcc")
    assert "mcc" in comparison.lower()
    top = ReportFormatter.format_top_features(
        np.array([0.2, 0.8]), ["a", "b"], np.array([[0.0, 1.0], [1.0, 3.0]]), np.array([0, 1])
    )
    assert top.index("  1. b") < top.index("  2. a")
    none = ReportFormatter.format_top_features(
        np.array([np.nan, np.nan]), ["a", "b"], np.array([[0.0, 1.0], [1.0, 3.0]]), np.array([0, 1])
    )
    assert "not available" in none
    missed = ReportFormatter.format_misclassified_samples(
        np.array([1, 0]), np.array([0, 0]), np.array([0, 1]), pd.Series(["geN|S1", "geN|S2"])
    )
    assert "S1" in missed
    unsupervised = {
        "comp": {"silhouette": {"score": 0.1}, "permanova": {"statistic": 1.0, "p_value": 0.5}}
    }
    assert "0.500" in ReportFormatter.format_unsupervised_metrics(unsupervised)


def test_the_logger_writes_a_file_and_closes_it(tmp_path):
    logger = Logger.setup_logger(name="phantom-test", log_dir=str(tmp_path), log_filename="a.log")
    Logger.log_tqdm(logger, "hello")
    file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    Logger.close(logger)
    assert logger.handlers == []
    assert file_handler.stream is None  # closed, not just dropped
    assert "hello" in (tmp_path / "a.log").read_text()


def test_result_plots_are_written(tmp_path):
    ResultPlots.plot_roc_curve(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.9, 0.4, 0.6]),
        save_path=str(tmp_path / "roc.pdf"),
    )
    coords = {
        m: np.random.default_rng(0).normal(size=(4, 2)) for m in ("comp", "host", "func", "fused")
    }
    ResultPlots.plot_unsupervised_grid(
        coords, [0, 1, 0, 1], ["S1", "S2", "S3", "S4"], save_path=str(tmp_path / "mds.pdf")
    )
    assert (tmp_path / "roc.pdf").stat().st_size > 0 and (tmp_path / "mds.pdf").stat().st_size > 0


# experiment figures


def _run(
    root: Path,
    name: str,
    metadata: dict,
    validators=("loocv",),
    wrong=("S1",),
    importance_dirs=("",),
):
    run = root / name
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps(metadata))
    for v in validators:
        d = run / v
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"mcc": {"score": 0.5}, "f1": {"score": 0.75}}))
        (d / "confusion_matrix.json").write_text(json.dumps({"TP": 2, "TN": 1, "FP": 1, "FN": 0}))
        ids = ["geN|S1", "geN|S2", "geN|S10"]
        pd.DataFrame(
            {
                "y_true": [1, 0, 1],
                "y_pred": [
                    0 if i.endswith(wrong) else t for i, t in zip(ids, [1, 0, 1], strict=True)
                ],
                "sample_id": ids,
            }
        ).to_csv(d / "predictions.csv", index=False)
        for sub in importance_dirs:
            (d / sub).mkdir(exist_ok=True)
            pd.DataFrame(
                {
                    "feature": ["lysis_ratio", "Escherichia"],
                    "importance": [0.6, 0.2],
                    "mean_class_0": [0.1, 0.5],
                    "mean_class_1": [0.3, 0.5],
                }
            ).to_csv(d / sub / "feature_importance.csv", index=False)
    return run


@pytest.fixture
def results(tmp_path):
    single, multi = tmp_path / "single-modal", tmp_path / "multi-modal"
    old = {"model_type": "rf", "vtool": "geN", "modality": "comp"}
    _run(single, "run_rf_geN_comp_20260101_000000", old)
    _run(single, "run_rf_geN_comp_20260102_000000", old, validators=("loocv", "rcv"), wrong=())
    _run(single, "run_catboost_geN_comp_20260103_000000", {**old, "model_type": "catboost"})
    _run(
        single,
        "run_rf_VIB_func_20260101_000000",
        {"model_type": "rf", "vtool": "VIB", "modality": "func"},
    )
    # A multimodal run from before the metadata recorded the tool.
    _run(
        multi,
        "run_early_fusion_20260101_000000",
        {"model_type": "rf", "fusion": "early", "comp": "/x/geN_PGN_FEAT.csv"},
        importance_dirs=("comp", "func"),
    )
    (single / "not_a_run").mkdir()
    return {"single": single, "multi": multi}


def test_the_newest_run_of_the_model_is_taken_per_setting(results):
    runs = figures.collect_runs(results["single"], "single", "rf")
    assert sorted(runs) == [("VIB", "func"), ("geN", "comp")]
    assert runs[("geN", "comp")].timestamp == "20260102_000000"
    multi = figures.collect_runs(results["multi"], "multi", "rf")
    assert list(multi) == [("geN", "early")]
    assert figures.collect_runs(results["single"].parent / "absent", "single", "rf") == {}


def test_the_metrics_table_is_tidy(results):
    runs = figures.collect_runs(results["single"], "single", "rf")
    table = figures.metrics_table(runs, ["loocv", "rcv"])
    assert set(table["metric"]) == {"mcc", "f1"}
    assert len(table[(table["vtool"] == "geN") & (table["validator"] == "rcv")]) == 2
    assert table[table["vtool"] == "VIB"]["validator"].unique().tolist() == ["loocv"]


def test_misclassification_rates(results):
    run = figures.collect_runs(results["single"], "single", "rf")[("VIB", "func")]
    rates = figures.misclassification_rates(run.path / "loocv" / "predictions.csv")
    assert rates.to_dict() == {"S1": 1.0, "S10": 0.0, "S2": 0.0}
    assert figures.misclassification_rates(run.path / "missing.csv") is None


def test_make_figures_writes_every_figure_that_has_data(results, tmp_path):
    output = tmp_path / "figures"
    written = figures.make_figures(results, "rf", ["loocv", "rcv"], LABELS, output)
    names = sorted(p.name for p in written)
    assert "metrics.csv" in names
    for kind, validator in (("single", "loocv"), ("single", "rcv"), ("multi", "loocv")):
        for figure in ("metrics", "confusion_matrices", "misclassified", "top_features"):
            assert f"{figure}_{kind}_{validator}.pdf" in names
    assert "metrics_multi_rcv.pdf" not in names  # no multimodal run has rcv


def test_a_model_without_importances_gets_an_empty_panel(tmp_path):
    path = tmp_path / "feature_importance.csv"
    pd.DataFrame(
        {"feature": ["a"], "importance": [float("nan")], "mean_class_0": [0], "mean_class_1": [1]}
    ).to_csv(path, index=False)
    assert figures._top_features(path).empty
    assert figures._top_features(tmp_path / "missing.csv") is None


def test_make_figures_needs_at_least_one_run(results, tmp_path):
    with pytest.raises(FileNotFoundError, match="no xgb classification runs"):
        figures.make_figures(results, "xgb", ["loocv"], LABELS, tmp_path / "out")


def test_display_labels():
    labels = figures.Labels.from_config(
        {
            "virus-identification-tools": {"geN": "genomad", "X": "newtool"},
            "modalities": {"comp": {"label": "Virus taxonomy"}},
        }
    )
    assert labels.virus_identification_tools == {"geN": "geNomad", "X": "newtool"}
    assert labels.groups("single") == {"comp": "Virus taxonomy"}
    assert labels.groups("multi") == figures.FUSION_NAMES
    assert figures._feature_label("defense_accessory_ratio") == "Defense accessory"
    assert figures._feature_label("Bacteroides") == "Bacteroides"


def test_text_is_escaped_only_when_latex_renders_it(monkeypatch):
    import matplotlib as mpl

    monkeypatch.setitem(mpl.rcParams, "text.usetex", False)
    assert figures._tex("a_b & c") == "a_b & c"
    monkeypatch.setitem(mpl.rcParams, "text.usetex", True)
    assert figures._tex("a_b & c") == r"a\_b \& c"
