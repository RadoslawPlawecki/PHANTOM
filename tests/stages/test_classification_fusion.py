"""classification.multi / classification.mds: grid, validation and the exact
config values handed to the (faked) executors."""

from __future__ import annotations

from pathlib import Path

import pytest

from phantom.runner.loader import RunFileError
from tests.stages.conftest import VERSION, fake_executor, jobs_for, run_stage, validate_stage


@pytest.fixture
def fake_multi(monkeypatch):
    import phantom.analysis.classification.experiments.multi_modal as mod

    return fake_executor(monkeypatch, mod, "MultiModalClassifier")


@pytest.fixture
def fake_mds(monkeypatch):
    import phantom.analysis.classification.experiments.mds as mod

    return fake_executor(monkeypatch, mod, "MdsAnalysis")


def test_multi_grid_defaults_to_all_vtools(context):
    jobs = jobs_for("classification.multi", context, version=VERSION, fusion=["early", "late"])
    assert [(j.params.vtool, j.params.fusion) for j in jobs] == [
        ("geN", "early"),
        ("geN", "late"),
        ("VIB", "early"),
        ("VIB", "late"),
        ("VS2", "early"),
        ("VS2", "late"),
    ]


@pytest.mark.parametrize(
    ("params", "expected_opt"),
    [
        ({"fusion": "early"}, (False, 0)),
        ({"fusion": "late"}, (False, 0)),
        ({"fusion": "late", "optimize_weights": True, "n_trials": 7}, (True, 7)),
    ],
)
def test_multi_passes_the_expected_config(ws, context, fake_multi, params, expected_opt):
    run_stage("classification.multi", context, version=VERSION, vtool="VIB", model="rf", **params)
    (run_config,) = fake_multi
    assert (run_config.opt, run_config.n_trials) == expected_opt
    assert Path(run_config.comp).name == "VIB_PGN_FEAT.csv"
    assert Path(run_config.host).name == "VIB_CHR_FEAT.csv"
    assert Path(run_config.func).name == "VIB_PHA_ORFs_PHV_FEAT.csv"
    assert Path(run_config.out_dir) == ws.root / "results" / "multi-modal" / VERSION
    assert (run_config.run_loocv, run_config.run_repeated, run_config.model_type) == (
        True,
        False,
        "rf",
    )


def test_multi_rejects_weight_tuning_for_early_fusion(context):
    with pytest.raises(RunFileError, match="optimize_weights applies to fusion = 'late' only"):
        jobs_for(
            "classification.multi", context, version=VERSION, fusion="early", optimize_weights=True
        )


def test_multi_reports_missing_modality_file(ws, context):
    (ws.stage_dir("extracted", "phavip") / "geN_PHA_ORFs_PHV_FEAT.csv").unlink()
    with pytest.raises(RunFileError, match="No geN PHV feature file"):
        validate_stage("classification.multi", context, version=VERSION, vtool="geN")


def test_mds_maps_files_and_default_out_dir(ws, context, fake_mds):
    run_stage("classification.mds", context, version=VERSION, vtool="geN")
    (run_config,) = fake_mds
    assert [Path(getattr(run_config, m)).name for m in ("comp", "host", "func")] == [
        "geN_PGN_FEAT.csv",
        "geN_CHR_FEAT.csv",
        "geN_PHA_ORFs_PHV_FEAT.csv",
    ]
    assert Path(run_config.out_dir) == ws.root / "results" / "mds" / VERSION
