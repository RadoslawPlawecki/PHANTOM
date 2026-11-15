"""classification.figures: which runs it reads, what it checks, where it writes."""

from __future__ import annotations

import json

import pytest

import phantom.stages  # noqa: F401 -- registers the stages
from phantom.runner.loader import RunFileError
from tests.stages.conftest import VERSION, jobs_for, run_stage, validate_stage


def _single_run(ws, model="rf", virus_identification_tool="geN", modality="comp"):
    run = (
        ws.root
        / "results"
        / "single-modal"
        / VERSION
        / f"run_{model}_{virus_identification_tool}_{modality}_20260101_000000"
    )
    (run / "loocv").mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps({"model_type": model, "vtool": virus_identification_tool, "modality": modality})
    )
    (run / "loocv" / "metrics.json").write_text(json.dumps({"mcc": {"score": 0.4}}))
    return run


def test_figures_need_runs_of_the_model(ws, context):
    with pytest.raises(RunFileError, match="no rf classification runs"):
        validate_stage("classification.figures", context, version=VERSION, model="rf")


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"validators": []}, "give at least one of loocv, rcv"),
        ({"kinds": []}, "give at least one of single, multi"),
        ({"version": "0.0"}, "version '0.0' not in config.toml"),
    ],
)
def test_figures_validation(context, params, message):
    with pytest.raises(RunFileError, match=message):
        jobs_for("classification.figures", context, **{"version": VERSION, "model": "rf", **params})


def test_figures_are_written_next_to_the_results(ws, context):
    _single_run(ws)
    (output,) = run_stage(
        "classification.figures", context, version=VERSION, model="rf", kinds=["single"]
    )
    assert output == ws.root / "results" / "figures" / VERSION / "rf"
    assert (output / "metrics.csv").is_file() and (output / "metrics_single_loocv.pdf").is_file()
    assert (output / "run.toml").is_file()


def test_figures_read_the_classification_results(ws, context):
    (job,) = jobs_for("classification.figures", context, version=VERSION, model="rf")
    inputs = job.stage.inputs(job.params, context)
    assert [params.name for params in inputs] == [VERSION, VERSION]
    assert {params.parent.name for params in inputs} == {"single-modal", "multi-modal"}
    assert job.stage.label(job.params) == "rf / single + multi"
