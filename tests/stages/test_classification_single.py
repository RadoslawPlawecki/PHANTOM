"""classification.single: validation against config/data, job grid, and the
mapping from run-file parameters to SingleModalConfig (real classifier faked)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from phantom.runner.loader import RunFileError, build_jobs, parse_run_doc
from phantom.runner.planner import execute, plan
from phantom.runner.stage import RunContext
from phantom.runner.stamps import StampStore
from tests._helpers.frames import extracted_matrix

VERSION = "9.9"


@pytest.fixture
def ws(tmp_workspace):
    matrix = extracted_matrix([("geN|S1", 1, 0), ("geN|S2", 0, 1)], ["t1", "t2"])
    for virus_identification_tool in ("geN", "VIB", "VS2"):
        tmp_workspace.write_extracted("cherry", f"{virus_identification_tool}_CHR_FEAT.csv", matrix)
    metadata = tmp_workspace.data_root / "genomes" / "genome_metadata.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["S1", "S2"], "label": [1, 0]}).to_csv(metadata, sep=";", index=False)
    return tmp_workspace


@pytest.fixture
def context(ws) -> RunContext:
    return RunContext.load(ws.config_path)


@pytest.fixture
def fake_classifier(monkeypatch):
    """Replaces SingleModalClassifier; records configs, creates the exp dir."""
    import phantom.analysis.classification.experiments.single_modal as mod

    seen = []

    class Saver:
        def __init__(self, exp_dir):
            self.exp_dir = exp_dir

    class FakeClassifier:
        def __init__(self, config):
            self.config = config
            self.saver = None

        def run(self):
            name = f"run_{self.config.model_type}_{self.config.vtool}_{self.config.seed}"
            exp = Path(self.config.out_dir) / name
            exp.mkdir(parents=True)
            self.saver = Saver(str(exp))
            seen.append(self.config)

    monkeypatch.setattr(mod, "SingleModalClassifier", FakeClassifier)
    return seen


def _jobs(step, context):
    """Build + data checks, as ``phantom validate`` does."""
    jobs = build_jobs(
        parse_run_doc({"steps": [{"stage": "classification.single", **step}]}), context
    )
    plan(jobs, context, StampStore(context.section_path("stamps")), force=True)
    return jobs


def test_vtool_defaults_to_every_configured_tool(context):
    jobs = _jobs({"version": VERSION, "modality": "host"}, context)
    assert [j.params.vtool for j in jobs] == ["geN", "VIB", "VS2"]
    assert jobs[0].label == "host / geN / catboost"


def test_grid_over_models(context):
    jobs = _jobs(
        {"version": VERSION, "modality": "host", "vtool": "geN", "model": ["rf", "xgb"]}, context
    )
    assert [j.params.model for j in jobs] == ["rf", "xgb"]


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"version": "1.0", "modality": "host"}, "version '1.0' not in config.toml"),
        ({"version": VERSION, "modality": "taxonomy"}, "modality 'taxonomy' not in [modalities]"),
        (
            {"version": VERSION, "modality": "host", "vtool": "XYZ"},
            "vtool 'XYZ' is not a virus identification tool",
        ),
        ({"version": VERSION, "modality": "host", "validators": []}, "at least one"),
        ({"version": VERSION, "modality": "comp"}, "No geN PGN feature file"),
    ],
)
def test_validation_errors(context, step, message):
    with pytest.raises(RunFileError) as exc:
        _jobs(step, context)
    assert any(message in e for e in exc.value.errors), exc.value.errors


def test_missing_metadata_is_reported(ws, context):
    (ws.data_root / "genomes" / "genome_metadata.csv").unlink()
    with pytest.raises(RunFileError, match="metadata with class labels not found"):
        _jobs({"version": VERSION, "modality": "host", "vtool": "geN"}, context)


def test_run_maps_params_to_classifier_config(ws, context, fake_classifier):
    step = {
        "version": VERSION,
        "modality": "host",
        "vtool": "VIB",
        "model": "rf",
        "validators": ["rcv"],
        "smote": True,
    }
    run = parse_run_doc({"steps": [{"stage": "classification.single", **step}]})
    store = StampStore(context.section_path("stamps"))
    planned = plan(build_jobs(run, context), context, store)
    (output,) = execute(planned, context, store, run)

    (run_config,) = fake_classifier
    assert Path(run_config.in_file).name == "VIB_CHR_FEAT.csv"
    assert Path(run_config.out_dir) == ws.root / "results" / "single-modal" / VERSION
    assert (run_config.modality, run_config.vtool, run_config.model_type) == ("host", "VIB", "rf")
    assert (run_config.run_loocv, run_config.run_repeated, run_config.use_smote) == (
        False,
        True,
        True,
    )
    assert (output / "run.toml").is_file()
    assert run_config.seed == 42


def test_a_seed_grid_gives_one_job_per_seed(ws, context, fake_classifier):
    step = {"version": VERSION, "modality": "host", "vtool": "VIB", "model": "rf", "seed": [1, 2]}
    run = parse_run_doc({"steps": [{"stage": "classification.single", **step}]})
    store = StampStore(context.section_path("stamps"))
    execute(plan(build_jobs(run, context), context, store), context, store, run)
    assert [run_config.seed for run_config in fake_classifier] == [1, 2]


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([1, 1], r"needs cases \(1\) and controls \(0\)"),
        ([1, 2], "Labels must be one of"),
    ],
)
def test_labels_unfit_for_classification_are_reported(ws, context, labels, message):
    metadata = ws.data_root / "genomes" / "genome_metadata.csv"
    pd.DataFrame({"id": ["S1", "S2"], "label": labels}).to_csv(metadata, sep=";", index=False)
    with pytest.raises(RunFileError, match=message):
        _jobs({"version": VERSION, "modality": "host", "vtool": "geN"}, context)


def test_a_tool_can_be_named_in_full(context):
    full = _jobs(
        {"version": VERSION, "modality": "host", "vtool": ["genomad", "VirSorter2"]}, context
    )
    short = _jobs({"version": VERSION, "modality": "host", "vtool": ["geN", "VS2"]}, context)
    assert [j.params for j in full] == [j.params for j in short]  # one job, one stamp
    assert [j.params.vtool for j in full] == ["geN", "VS2"]
