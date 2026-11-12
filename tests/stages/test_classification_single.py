"""classification.single: validation against config/data, job grid, and the
mapping from run-file parameters to SingleOmicConfig (real classifier faked)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests._helpers.frames import extracted_matrix

from phantom.runner.loader import RunFileError, build_jobs, parse_run_doc
from phantom.runner.planner import DONE, RUN, execute, plan
from phantom.runner.stage import RunContext
from phantom.runner.stamps import StampStore

VERSION = "9.9"


@pytest.fixture
def ws(tmp_workspace):
    matrix = extracted_matrix([("geN|S1", 1, 0), ("geN|S2", 0, 1)], ["t1", "t2"])
    for vtool in ("geN", "VIB", "VS2"):
        tmp_workspace.write_extracted("cherry", f"{vtool}_CHR_FEAT.csv", matrix)
    meta = tmp_workspace.data_root / "genomes" / "genome_metadata.csv"
    meta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["S1", "S2"], "label": [1, 0]}).to_csv(meta, sep=";", index=False)
    return tmp_workspace


@pytest.fixture
def ctx(ws) -> RunContext:
    return RunContext.load(ws.config_path)


@pytest.fixture
def fake_classifier(monkeypatch):
    """Replaces SingleOmicClassifier; records configs, creates the exp dir."""
    import phantom.analysis.classification.execution.single_omic as mod

    seen = []

    class Saver:
        def __init__(self, exp_dir):
            self.exp_dir = exp_dir

    class FakeClassifier:
        def __init__(self, config):
            self.config = config
            self.saver = None

        def run(self):
            exp = Path(self.config.out_dir) / f"run_{self.config.model_type}_{self.config.vtool}"
            exp.mkdir(parents=True)
            self.saver = Saver(str(exp))
            seen.append(self.config)

    monkeypatch.setattr(mod, "SingleOmicClassifier", FakeClassifier)
    return seen


def _jobs(step, ctx):
    """Build + data checks, as ``phantom validate`` does."""
    jobs = build_jobs(parse_run_doc({"steps": [{"stage": "classification.single", **step}]}), ctx)
    plan(jobs, ctx, StampStore(ctx.section_path("stamps")), force=True)
    return jobs


def test_vtool_defaults_to_every_configured_tool(ctx):
    jobs = _jobs({"version": VERSION, "modality": "host"}, ctx)
    assert [j.params.vtool for j in jobs] == ["geN", "VIB", "VS2"]
    assert jobs[0].label == "host / geN / catboost"


def test_grid_over_models(ctx):
    jobs = _jobs(
        {"version": VERSION, "modality": "host", "vtool": "geN", "model": ["rf", "xgb"]}, ctx
    )
    assert [j.params.model for j in jobs] == ["rf", "xgb"]


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"version": "1.0", "modality": "host"}, "version '1.0' not in config.toml"),
        ({"version": VERSION, "modality": "taxonomy"}, "modality 'taxonomy' not in [modalities]"),
        ({"version": VERSION, "modality": "host", "vtool": "XYZ"}, "vtool 'XYZ' not in [tools]"),
        ({"version": VERSION, "modality": "host", "validators": []}, "at least one"),
        ({"version": VERSION, "modality": "comp"}, "No geN PGN feature file"),
    ],
)
def test_validation_errors(ctx, step, message):
    with pytest.raises(RunFileError) as exc:
        _jobs(step, ctx)
    assert any(message in e for e in exc.value.errors), exc.value.errors


def test_missing_metadata_is_reported(ws, ctx):
    (ws.data_root / "genomes" / "genome_metadata.csv").unlink()
    with pytest.raises(RunFileError, match="metadata with class labels not found"):
        _jobs({"version": VERSION, "modality": "host", "vtool": "geN"}, ctx)


def test_run_maps_params_to_classifier_config(ws, ctx, fake_classifier):
    step = {
        "version": VERSION,
        "modality": "host",
        "vtool": "VIB",
        "model": "rf",
        "validators": ["rcv"],
        "smote": True,
    }
    run = parse_run_doc({"steps": [{"stage": "classification.single", **step}]})
    store = StampStore(ctx.section_path("stamps"))
    planned = plan(build_jobs(run, ctx), ctx, store)
    (out,) = execute(planned, ctx, store, run)

    (cfg,) = fake_classifier
    assert Path(cfg.in_file).name == "VIB_CHR_FEAT.csv"
    assert Path(cfg.out_dir) == ws.root / "results" / "single-omic" / VERSION
    assert (cfg.modality, cfg.vtool, cfg.model_type) == ("host", "VIB", "rf")
    assert (cfg.run_loocv, cfg.run_repeated, cfg.use_smote) == (False, True, True)
    assert (out / "run.toml").is_file()


def test_relabeling_invalidates_results(ws, ctx, fake_classifier):
    run = parse_run_doc(
        {
            "steps": [
                {
                    "stage": "classification.single",
                    "version": VERSION,
                    "modality": "host",
                    "vtool": "geN",
                }
            ]
        }
    )
    store = StampStore(ctx.section_path("stamps"))
    execute(plan(build_jobs(run, ctx), ctx, store), ctx, store, run)
    assert plan(build_jobs(run, ctx), ctx, store)[0].state == DONE

    meta = ws.data_root / "genomes" / "genome_metadata.csv"
    pd.DataFrame({"id": ["S1", "S2"], "label": [0, 1]}).to_csv(meta, sep=";", index=False)
    (pj,) = plan(build_jobs(run, ctx), ctx, store)
    assert (pj.state, pj.reason.startswith("inputs changed")) == (RUN, True)
