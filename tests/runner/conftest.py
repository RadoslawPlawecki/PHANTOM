"""Toy stages for exercising the runner without any real pipeline code.

``test.echo`` reads ``<root>/in.txt`` and writes one directory per job under
``<root>/out``; ``test.consume`` reads ``<root>/out`` -- so a pending echo job
must make a later consume job stale. Both record their calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from phantom.runner.params import param
from phantom.runner.stage import REGISTRY, RunContext, Stage


@dataclass(frozen=True)
class EchoParams:
    name: str = param(help="required name", grid=True)
    n: int = param(1, grid=True)
    tags: list[Literal["a", "b"]] = param(default_factory=list)
    flag: bool = param(False)
    src: str | None = param(None)


class Echo(Stage):
    name = "test.echo"
    summary = "toy producer"
    Params = EchoParams

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[EchoParams] = []
        self.fail_on: str | None = None

    def inputs(self, p, ctx):
        return [ctx.resolve(p.src) if p.src else self.root / "in.txt"]

    def writes(self, p, ctx):
        return [self.root / "out"]

    def run(self, p, ctx):
        if p.name == self.fail_on:
            raise RuntimeError("boom")
        out = self.root / "out" / f"{p.name}_{p.n}"
        out.mkdir(parents=True, exist_ok=True)
        self.calls.append(p)
        return [out]

    def label(self, p):
        return f"{p.name}/{p.n}"


@dataclass(frozen=True)
class ConsumeParams:
    tag: str = param("x")


class Consume(Stage):
    name = "test.consume"
    summary = "toy consumer"
    Params = ConsumeParams

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[ConsumeParams] = []

    def inputs(self, p, ctx):
        return [self.root / "out"]

    def check_inputs(self, p, ctx):
        # Needs at least one echo output to consume.
        out = self.root / "out"
        return [] if any(out.iterdir()) else [f"nothing to consume in {out}"]

    def writes(self, p, ctx):
        return [self.root / "final"]

    def run(self, p, ctx):
        out = self.root / "final" / p.tag
        out.mkdir(parents=True, exist_ok=True)
        self.calls.append(p)
        return [out]


@pytest.fixture
def ctx(tmp_workspace) -> RunContext:
    return RunContext.load(tmp_workspace.config_path)


@pytest.fixture
def toy(tmp_workspace):
    root = tmp_workspace.root / "toy"
    (root / "out").mkdir(parents=True)
    (root / "in.txt").write_text("v1", encoding="utf-8")
    echo, consume = Echo(root), Consume(root)
    REGISTRY[echo.name] = echo
    REGISTRY[consume.name] = consume
    yield echo, consume, root
    REGISTRY.pop(echo.name, None)
    REGISTRY.pop(consume.name, None)
