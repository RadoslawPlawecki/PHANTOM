"""features.optimize's mapping onto FeatureOptimizer.optimize (the search is faked)."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.stages.conftest import VERSION, jobs_for, run_stage, validate_stage

import phantom.stages  # noqa: F401 -- registers the stages
from phantom.runner.loader import RunFileError

# --- features.optimize --------------------------------------------------------------


@pytest.fixture
def fake_optimize(ws, monkeypatch):
    import phantom.analysis.features.optimization.optimizer as opt
    import phantom.analysis.features.optimization.search_core as core

    merged = ws.stage_dir("raw_merged", "phagcn")
    merged.mkdir(parents=True)
    pd.DataFrame({"Accession": ["S1_k1"]}).to_csv(merged / "geN_PGN_M.csv", sep=";", index=False)
    calls = []

    def optimize(tool, mode, config, out_dir, source, df, **knobs):
        out_dir.mkdir(parents=True, exist_ok=True)
        calls.append(
            {"tool": tool, "mode": mode, "config": config, "out": out_dir, "src": source, **knobs}
        )

    monkeypatch.setattr(opt, "run_optimization", optimize)
    monkeypatch.setattr(core, "load_masked", lambda tool, path, mask: pd.DataFrame())
    return calls


@pytest.mark.parametrize(
    ("mode", "nested", "permutation"),
    [
        ("exploratory", False, False),
        ("nested", True, False),
        ("permutation", False, True),
        ("all", True, True),
    ],
)
def test_optimize_records_only_the_knobs_its_mode_uses(
    ws, ctx, fake_optimize, mode, nested, permutation
):
    run_stage(
        "features.optimize", ctx, version=VERSION, tool="phagcn", vtool="geN", mode=mode, model="rf"
    )
    (call,) = fake_optimize
    assert (call["outer_folds"] is not None) is nested
    assert (call["top_k_features"] is not None) is nested
    assert (call["n_permutations"] is not None) is permutation
    assert call["config"].model == "rf"
    assert call["src"].name == "geN_PGN_M.csv"
    assert call["out"] == ws.root / "results" / "optimization" / VERSION / "phagcn" / "geN"


def test_optimize_validation(ws, ctx, fake_optimize):
    with pytest.raises(RunFileError, match="outer_folds must be >= 2"):
        jobs_for(
            "features.optimize", ctx, version=VERSION, tool="phagcn", vtool="geN", outer_folds=1
        )
    with pytest.raises(RunFileError, match=r"no VIB_\*\.csv merged file"):
        validate_stage("features.optimize", ctx, version=VERSION, tool="phagcn", vtool="VIB")
