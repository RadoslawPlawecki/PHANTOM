"""features.diversity / features.summarize_functions adapters (the analyses
themselves are faked; their outputs are covered by the regression harness)."""

from __future__ import annotations

import pytest
from tests.stages.conftest import VERSION, jobs_for, run_stage, validate_stage

from phantom.runner.loader import RunFileError


def test_diversity_grid_and_default_out_dir(ws, ctx, monkeypatch):
    import phantom.analysis.features.diversity as mod

    calls = []

    class FakeAnalyzer:
        def __init__(self, version, config_mgr=None):
            self.version = version

        def run(self, out_dir, ratio_mode="ratio_of_means"):
            out_dir.mkdir(parents=True, exist_ok=True)
            calls.append((self.version, out_dir, ratio_mode))

    monkeypatch.setattr(mod, "DiversityAnalyzer", FakeAnalyzer)
    run_stage(
        "features.diversity", ctx, version=VERSION, ratio_mode=["ratio_of_means", "mean_of_ratios"]
    )
    out = ws.root / "results" / "diversity" / VERSION
    assert calls == [(VERSION, out, "ratio_of_means"), (VERSION, out, "mean_of_ratios")]


def test_diversity_rejects_unknown_ratio_mode(ctx):
    with pytest.raises(RunFileError, match="ratio_mode: expected one of"):
        jobs_for("features.diversity", ctx, version=VERSION, ratio_mode="median")


def test_diversity_needs_preprocessed_data(ws, ctx):
    import shutil

    for tool in ("phagcn", "cherry"):
        shutil.rmtree(ws.stage_dir("preprocessed", tool))
    with pytest.raises(RunFileError, match="no preprocessed phagcn/cherry data"):
        validate_stage("features.diversity", ctx, version=VERSION)


def test_summarize_validates_inputs(ws, ctx):
    # The fixture config has no [phanotate] section.
    with pytest.raises(RunFileError, match=r"config.toml is missing \[phanotate\].path"):
        jobs_for("features.summarize_functions", ctx, version=VERSION)
