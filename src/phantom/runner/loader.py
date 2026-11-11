"""
Reading run files into validated jobs.

A run file is TOML with an optional ``[run]`` table (``name``,
``description``), one ``[[steps]]`` table per step (``stage = "..."`` plus
that stage's parameters) and, in files written by PHANTOM itself, a
``[provenance]`` table that is informational only. Each step expands into one
or more :class:`Job` s: a list given for a ``grid=True`` parameter becomes
one job per value.

All problems are collected and raised together as :class:`RunFileError`, so
a user sees every mistake in one go rather than one per attempt.
"""

from __future__ import annotations

import difflib
import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phantom.runner import params as P
from phantom.runner.stage import RunContext, Stage, get_stage

TOP_LEVEL_KEYS = {"run", "steps", "provenance"}


class RunFileError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass
class Step:
    index: int  # 1-based, as shown to the user
    stage: str
    values: dict[str, Any]


@dataclass
class RunFile:
    path: Path | None
    meta: dict[str, Any]
    steps: list[Step]

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or (self.path.stem if self.path else "run"))


@dataclass
class Job:
    step: Step
    stage: Stage
    params: Any

    @property
    def label(self) -> str:
        return self.stage.label(self.params)


def load_run_file(path: Path | str) -> RunFile:
    path = Path(path)
    try:
        with open(path, "rb") as f:
            doc = tomllib.load(f)
    except FileNotFoundError:
        raise RunFileError([f"Run file not found: {path}"]) from None
    except tomllib.TOMLDecodeError as e:
        raise RunFileError([f"{path}: invalid TOML: {e}"]) from None
    return parse_run_doc(doc, path)


def parse_run_doc(doc: dict[str, Any], path: Path | None = None) -> RunFile:
    errors = [
        f"unknown top-level table [{key}] (allowed: {', '.join(sorted(TOP_LEVEL_KEYS))})"
        for key in doc
        if key not in TOP_LEVEL_KEYS
    ]
    raw_steps = doc.get("steps", [])
    if not isinstance(raw_steps, list) or not raw_steps:
        errors.append("no [[steps]] defined")
        raw_steps = []
    steps = []
    for i, raw in enumerate(raw_steps, start=1):
        values = dict(raw)
        stage = values.pop("stage", None)
        if not isinstance(stage, str):
            errors.append(f"step {i}: missing 'stage = \"...\"'")
            continue
        steps.append(Step(index=i, stage=stage, values=values))
    if errors:
        raise RunFileError(errors)
    return RunFile(path=path, meta=dict(doc.get("run", {})), steps=steps)


def parse_override(text: str) -> tuple[str | None, str, Any]:
    """
    ``key=value`` (applies to every step that has ``key``) or
    ``stage.name.key=value`` (only steps of that stage). The value is parsed
    as TOML (``3``, ``true``, ``["a", "b"]``), falling back to a plain string.
    """
    if "=" not in text:
        raise RunFileError([f"--set {text!r}: expected key=value"])
    lhs, raw = text.split("=", 1)
    lhs = lhs.strip()
    stage, key = lhs.rsplit(".", 1) if "." in lhs else (None, lhs)
    try:
        value = tomllib.loads(f"v = {raw.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return stage, key, value


def apply_overrides(run: RunFile, overrides: list[str]) -> RunFile:
    errors = []
    for text in overrides:
        stage_name, key, value = parse_override(text)
        hit = False
        for step in run.steps:
            if stage_name is not None and step.stage != stage_name:
                continue
            try:
                stage = get_stage(step.stage)
            except KeyError:
                continue
            if key in {f.name for f in P.fields(stage.Params)}:
                step.values[key] = value
                hit = True
        if not hit:
            where = f"stage {stage_name!r}" if stage_name else "any step"
            errors.append(f"--set {text!r}: no parameter {key!r} in {where} of this run file")
    if errors:
        raise RunFileError(errors)
    return run


def build_jobs(run: RunFile, ctx: RunContext) -> list[Job]:
    errors: list[str] = []
    jobs: list[Job] = []
    for step in run.steps:
        prefix = f"step {step.index} ({step.stage})"
        try:
            stage = get_stage(step.stage)
        except KeyError as e:
            errors.append(f"step {step.index}: {e.args[0]}")
            continue
        step_jobs, step_errors = _expand_step(step, stage, ctx)
        for job in step_jobs:
            step_errors += stage.validate(job.params, ctx)
        # A grid repeats the same mistake per job; each message already names
        # the offending value, so report it once.
        errors += [f"{prefix}: {msg}" for msg in dict.fromkeys(step_errors)]
        jobs += step_jobs
    if errors:
        raise RunFileError(errors)
    return jobs


def _expand_step(step: Step, stage: Stage, ctx: RunContext) -> tuple[list[Job], list[str]]:
    fields = {f.name: f for f in P.fields(stage.Params)}
    hints = P.type_hints(stage.Params)
    errors = []

    for key in step.values:
        if key not in fields:
            hint = difflib.get_close_matches(key, fields, n=1)
            errors.append(
                f"unknown parameter {key!r}" + (f" -- did you mean {hint[0]!r}?" if hint else "")
            )

    dynamic = stage.defaults(ctx)
    values: dict[str, Any] = {}
    for name, f in fields.items():
        if name in step.values:
            values[name] = step.values[name]
        elif name in dynamic:
            values[name] = dynamic[name]
        elif P.has_static_default(f):
            values[name] = P.static_default(f)
        else:
            errors.append(f"missing required parameter {name!r} ({P.describe(hints[name])})")

    # Type-check every given value (each element, for a grid list) so all
    # mistakes in a step are reported together.
    grid_keys = []
    for name, value in values.items():
        tp = hints[name]
        if P.is_grid(fields[name]) and isinstance(value, list):
            grid_keys.append(name)
            if not value:
                errors.append(f"{name}: empty list")
            errors += [
                f"{name}: expected {P.describe(tp)}, got {v!r}" for v in value if not P.check(v, tp)
            ]
        elif not P.check(value, tp):
            errors.append(f"{name}: expected {P.describe(tp)}, got {value!r}")
    if errors:
        return [], errors

    combos = itertools.product(*(values[k] for k in grid_keys)) if grid_keys else [()]
    jobs = [
        Job(
            step=step,
            stage=stage,
            params=stage.Params(**{**values, **dict(zip(grid_keys, combo, strict=True))}),
        )
        for combo in combos
    ]
    return jobs, errors
