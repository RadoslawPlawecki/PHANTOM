"""classification.multi / classification.mds: grid, validation and the exact
config values handed to the (faked) executors."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.stages.conftest import VERSION, fake_executor, jobs_for, run_stage, validate_stage

from phantom.runner.loader import RunFileError


@pytest.fixture
def fake_multi(monkeypatch):
    import phantom.analysis.classification.execution.multi_omic as mod

    return fake_executor(monkeypatch, mod, "MultiOmicClassifier")


@pytest.fixture
def fake_mds(monkeypatch):
    import phantom.analysis.classification.execution.unsupervised_classifier as mod

    return fake_executor(monkeypatch, mod, "UnsupervisedClassifier")


def test_multi_grid_defaults_to_all_vtools(ctx):
    jobs = jobs_for("classification.multi", ctx, version=VERSION, fusion=["early", "late"])
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
def test_multi_passes_the_menu_equivalent_config(ws, ctx, fake_multi, params, expected_opt):
    run_stage("classification.multi", ctx, version=VERSION, vtool="VIB", model="rf", **params)
    (cfg,) = fake_multi
    assert (cfg.opt, cfg.n_trials) == expected_opt
    assert Path(cfg.comp).name == "VIB_PGN_FEAT.csv"
    assert Path(cfg.host).name == "VIB_CHR_FEAT.csv"
    assert Path(cfg.func).name == "VIB_PHA_ORFs_PHV_FEAT.csv"
    assert Path(cfg.out_dir) == ws.root / "results" / "multi-omic" / VERSION
    assert (cfg.run_loocv, cfg.run_repeated, cfg.model_type) == (True, False, "rf")


def test_multi_rejects_weight_tuning_for_early_fusion(ctx):
    with pytest.raises(RunFileError, match="optimize_weights applies to fusion = 'late' only"):
        jobs_for(
            "classification.multi", ctx, version=VERSION, fusion="early", optimize_weights=True
        )


def test_multi_reports_missing_modality_file(ws, ctx):
    (ws.stage_dir("extracted", "phavip") / "geN_PHA_ORFs_PHV_FEAT.csv").unlink()
    with pytest.raises(RunFileError, match="No geN PHV feature file"):
        validate_stage("classification.multi", ctx, version=VERSION, vtool="geN")


def test_mds_maps_files_and_default_out_dir(ws, ctx, fake_mds):
    run_stage("classification.mds", ctx, version=VERSION, vtool="geN")
    (cfg,) = fake_mds
    assert [Path(getattr(cfg, m)).name for m in ("comp", "host", "func")] == [
        "geN_PGN_FEAT.csv",
        "geN_CHR_FEAT.csv",
        "geN_PHA_ORFs_PHV_FEAT.csv",
    ]
    assert Path(cfg.out_dir) == ws.root / "results" / "mds" / VERSION
