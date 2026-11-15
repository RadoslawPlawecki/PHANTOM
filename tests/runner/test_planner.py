"""Skip/rerun decisions, staleness propagation, provenance and failure handling."""

from __future__ import annotations

import os
import shutil

import pytest

from phantom.runner.loader import RunFileError, build_jobs, load_run_file, parse_run_doc
from phantom.runner.planner import DONE, RUN, JobFailed, execute, plan, resolve_from
from phantom.runner.stamps import StampStore

pytestmark = pytest.mark.critical


@pytest.fixture
def store(context):
    return StampStore(context.section_path("stamps"))


def _run(doc, context, store, **kw):
    run = parse_run_doc(doc)
    planned = plan(build_jobs(run, context), context, store, **kw)
    return run, planned


def _states(planned):
    return [planned_job.state for planned_job in planned]


ECHO = {"steps": [{"stage": "test.echo", "name": "a", "n": [1, 2]}]}


def test_first_run_then_up_to_date(toy, context, store):
    echo, _, _ = toy
    run, planned = _run(ECHO, context, store)
    assert _states(planned) == [RUN, RUN]
    execute(planned, context, store, run)
    assert [c.n for c in echo.calls] == [1, 2]

    _, planned = _run(ECHO, context, store)
    assert _states(planned) == [DONE, DONE]
    assert "up to date" in planned[0].reason


def test_changed_parameter_is_a_new_job(toy, context, store):
    run, planned = _run(ECHO, context, store)
    execute(planned, context, store, run)
    doc = {"steps": [{"stage": "test.echo", "name": "a", "n": [1, 2, 3]}]}
    _, planned = _run(doc, context, store)
    assert _states(planned) == [DONE, DONE, RUN]


def test_changed_input_reruns(toy, context, store):
    _, _, root = toy
    run, planned = _run(ECHO, context, store)
    execute(planned, context, store, run)
    (root / "in.txt").write_text("v2 -- different size", encoding="utf-8")
    _, planned = _run(ECHO, context, store)
    assert _states(planned) == [RUN, RUN]
    assert "inputs changed" in planned[0].reason


def test_rewrite_with_same_size_reruns(toy, context, store):
    _, _, root = toy
    run, planned = _run(ECHO, context, store)
    execute(planned, context, store, run)
    path = root / "in.txt"
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _, planned = _run(ECHO, context, store)
    assert _states(planned) == [RUN, RUN]


def test_deleted_output_reruns(toy, context, store):
    _, _, root = toy
    run, planned = _run(ECHO, context, store)
    execute(planned, context, store, run)
    shutil.rmtree(root / "out" / "a_1")
    _, planned = _run(ECHO, context, store)
    assert _states(planned) == [RUN, DONE]
    assert "previous output missing" in planned[0].reason


def test_force_reruns_everything(toy, context, store):
    run, planned = _run(ECHO, context, store)
    execute(planned, context, store, run)
    _, planned = _run(ECHO, context, store, force=True)
    assert _states(planned) == [RUN, RUN]


PIPE = {
    "steps": [
        {"stage": "test.echo", "name": "a"},
        {"stage": "test.consume"},
    ]
}


def test_pending_upstream_makes_downstream_stale(toy, context, store):
    run, planned = _run(PIPE, context, store)
    execute(planned, context, store, run)
    # Upstream param change: its job runs, and so must the consumer, even
    # though the consumer's stamp still matches today's (unchanged) inputs.
    doc = {"steps": [{"stage": "test.echo", "name": "b"}, {"stage": "test.consume"}]}
    _, planned = _run(doc, context, store)
    assert _states(planned) == [RUN, RUN]
    assert planned[1].reason == "input rewritten by job 1"


def test_only_and_from_select_steps(toy, context, store):
    _, _, root = toy
    (root / "out" / "earlier").mkdir()  # something for the consumer to read
    _, planned = _run(PIPE, context, store, only=["test.consume"])
    assert _states(planned) == ["deselected", RUN]
    _, planned = _run(PIPE, context, store, start=resolve_from(parse_run_doc(PIPE), "test.consume"))
    assert _states(planned) == ["deselected", RUN]
    with pytest.raises(ValueError, match="no such stage"):
        resolve_from(parse_run_doc(PIPE), "nope")


def test_failure_stops_and_does_not_stamp_failed_job(toy, context, store):
    echo, consume, _ = toy
    doc = {"steps": [{"stage": "test.echo", "name": ["ok", "bad"]}, {"stage": "test.consume"}]}
    echo.fail_on = "bad"
    run, planned = _run(doc, context, store)
    with pytest.raises(JobFailed, match=r"test\.echo -- bad/1"):
        execute(planned, context, store, run)
    assert consume.calls == []  # later jobs not run
    echo.fail_on = None
    _, planned = _run(doc, context, store)
    assert _states(planned) == [DONE, RUN, RUN]  # finished job skipped on retry


def test_provenance_run_toml_round_trips(toy, context, store):
    _, _, root = toy
    doc = {"run": {"name": "demo"}, "steps": [{"stage": "test.echo", "name": "a", "tags": ["b"]}]}
    run, planned = _run(doc, context, store)
    execute(planned, context, store, run)
    prov_file = root / "out" / "a_1" / "run.toml"
    assert prov_file.is_file()
    reloaded = load_run_file(prov_file)
    assert reloaded.metadata["name"] == "demo"
    (job,) = build_jobs(reloaded, context)
    assert job.params == planned[0].job.params
    # A provenance file names exactly the job it came from, so it is up to date.
    assert plan([job], context, store)[0].state == DONE


def test_data_checks_wait_for_an_upstream_producer(toy, context, store):
    _, consume, _ = toy
    # out/ is empty: on its own, the consumer is rejected before anything runs ...
    with pytest.raises(RunFileError, match="nothing to consume"):
        _run({"steps": [{"stage": "test.consume"}]}, context, store)
    # ... but after a producer in the same run it is deferred, then passes.
    run, planned = _run(PIPE, context, store)
    assert [planned_job.deferred for planned_job in planned] == [False, True]
    execute(planned, context, store, run)
    assert len(consume.calls) == 1


def test_deferred_check_failure_stops_before_the_job(toy, context, store):
    echo, consume, _ = toy

    def produce_nothing(p, c):
        return []  # writes nothing into out/

    echo.run = produce_nothing
    run, planned = _run(PIPE, context, store)
    with pytest.raises(JobFailed, match="nothing to consume"):
        execute(planned, context, store, run)
    assert consume.calls == []


def test_stamp_fingerprints_inputs_as_they_are_after_the_run(toy, context, store):
    run, planned = _run(PIPE, context, store)
    execute(planned, context, store, run)
    # The consumer's inputs did not exist at plan time; its stamp must match
    # them now, so a rerun skips it.
    _, planned = _run(PIPE, context, store)
    assert _states(planned) == [DONE, DONE]
