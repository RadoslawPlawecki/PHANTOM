"""
Planning and executing jobs.

``plan`` decides, per job, whether it runs or is skipped: selected out by
``--only``/``--from``, up to date according to its stamp, or forced. A job
that will run marks every *later* job reading from a directory it writes
into as needing to run too -- so changing an extraction parameter reruns
the classification built on it, while everything upstream stays skipped.

``execute`` runs the planned jobs in order, writes provenance into each
output and a stamp after each success, and stops at the first failure.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from phantom.runner.loader import Job, RunFile, RunFileError
from phantom.runner.provenance import git_state, phantom_version, write_run_toml
from phantom.runner.stage import RunContext
from phantom.runner.stamps import StampStore, Status

RUN, DONE, DESELECTED = "run", "done", "deselected"


@dataclass
class PlannedJob:
    number: int
    job: Job
    state: str
    reason: str
    inputs: list[Path]
    # Inputs produced by an earlier job of this run: their data checks wait
    # until just before this job runs.
    deferred: bool = False


def _selected(job: Job, only: list[str] | None, start: int | None) -> bool:
    if start is not None and job.step.index < start:
        return False
    if only:
        return job.step.stage in only or str(job.step.index) in only
    return True


def resolve_from(run: RunFile, value: str | None) -> int | None:
    """``--from`` accepts a step number or a stage name (its first step)."""
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    for step in run.steps:
        if step.stage == value:
            return step.index
    raise ValueError(f"--from {value!r}: no such stage in this run file")


def _overlaps(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def plan(
    jobs: list[Job],
    ctx: RunContext,
    store: StampStore,
    *,
    force: bool = False,
    only: list[str] | None = None,
    start: int | None = None,
    check_inputs: bool = True,
) -> list[PlannedJob]:
    """
    Decides per job whether it runs. With ``check_inputs``, the data checks
    of every job that runs and does not depend on an earlier job are done
    now, and all their failures are raised together as RunFileError.
    """
    planned: list[PlannedJob] = []
    will_write: list[tuple[int, Path]] = []
    for number, job in enumerate(jobs, start=1):
        inputs = job.stage.inputs(job.params, ctx)
        if not _selected(job, only, start):
            planned.append(PlannedJob(number, job, DESELECTED, "not selected", inputs))
            continue
        upstream = next((n for n, w in will_write if any(_overlaps(w, i) for i in inputs)), None)
        status: Status
        if force:
            status = Status(RUN, "forced (--force)")
        else:
            status = store.status(job.stage.name, job.params, inputs)
            if status.state == DONE and upstream is not None:
                status = Status(RUN, f"input rewritten by job {upstream}")
        if status.state == RUN:
            will_write += [(number, w) for w in job.stage.writes(job.params, ctx)]
        planned.append(
            PlannedJob(
                number,
                job,
                status.state,
                status.reason,
                inputs,
                deferred=status.state == RUN and upstream is not None,
            )
        )
    if check_inputs:
        errors = [
            f"job {pj.number} (step {pj.job.step.index}, {pj.job.stage.name}"
            f"{' -- ' + pj.job.label if pj.job.label else ''}): {msg}"
            for pj in planned
            if pj.state == RUN and not pj.deferred
            for msg in dict.fromkeys(_check(pj.job, ctx))
        ]
        if errors:
            raise RunFileError(errors)
    return planned


def _check(job: Job, ctx: RunContext) -> list[str]:
    try:
        return job.stage.check_inputs(job.params, ctx)
    except (KeyError, ValueError, FileNotFoundError) as e:
        return [str(e.args[0]) if e.args else str(e)]


def format_plan(planned: list[PlannedJob]) -> str:
    rows = [("#", "step", "stage", "job", "state")]
    for pj in planned:
        marker = {RUN: "RUN ", DONE: "skip", DESELECTED: "    "}[pj.state]
        rows.append(
            (
                str(pj.number),
                str(pj.job.step.index),
                pj.job.stage.name,
                pj.job.label,
                f"{marker} {pj.reason}" + (" (inputs from an earlier job)" if pj.deferred else ""),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = ["  ".join(r[i].ljust(widths[i]) for i in range(4)) + "  " + r[4] for r in rows]
    n_run = sum(pj.state == RUN for pj in planned)
    n_done = sum(pj.state == DONE for pj in planned)
    lines.append(f"\n{n_run} job(s) to run, {n_done} up to date.")
    return "\n".join(lines)


def execute(
    planned: list[PlannedJob],
    ctx: RunContext,
    store: StampStore,
    run: RunFile,
) -> list[Path]:
    """Runs every RUN job in order; raises on the first failure."""
    produced: list[Path] = []
    to_run = [pj for pj in planned if pj.state == RUN]
    for i, pj in enumerate(to_run, start=1):
        job = pj.job
        header = f"[{i}/{len(to_run)}] {job.stage.name}"
        print(f"\n=== {header}{' -- ' + job.label if job.label else ''} ===", flush=True)
        if pj.deferred:
            problems = _check(job, ctx)
            if problems:
                raise JobFailed(pj, "; ".join(dict.fromkeys(problems)))
        started = datetime.now()
        try:
            outputs = job.stage.run(job.params, ctx)
        except Exception as e:
            traceback.print_exc()
            raise JobFailed(pj, e) from e
        if outputs is None:  # deliberately produced nothing (dry run): no stamp
            print("[INFO] nothing written (dry run); the job stays pending")
            continue
        # Recomputed: inputs produced by an earlier job of this run did not
        # exist when the plan was made.
        inputs = job.stage.inputs(job.params, ctx)
        for out in outputs:
            if out.is_dir():
                write_run_toml(
                    out,
                    stage=job.stage.name,
                    params=job.params,
                    run_meta=run.meta,
                    source=run.path,
                    inputs=inputs,
                )
        store.write(
            job.stage.name,
            job.params,
            inputs,
            outputs,
            started,
            extra={"phantom_version": phantom_version(), **git_state()},
        )
        produced += outputs
        for out in outputs:
            print(f"[DONE] {out}")
    return produced


class JobFailed(Exception):
    def __init__(self, planned_job: PlannedJob, cause: Any):
        self.planned_job = planned_job
        self.cause = cause
        super().__init__(
            f"job {planned_job.number} ({planned_job.job.stage.name}"
            f"{' -- ' + planned_job.job.label if planned_job.job.label else ''}) failed: {cause}"
        )
