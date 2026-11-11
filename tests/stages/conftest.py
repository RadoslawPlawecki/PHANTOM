"""A workspace with extracted matrices for all three modalities x vtools,
preprocessed phagcn/cherry/phavip tables and a labeled metadata CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests._helpers.frames import extracted_matrix

from phantom.runner.loader import build_jobs, parse_run_doc
from phantom.runner.planner import execute, plan
from phantom.runner.stage import RunContext
from phantom.runner.stamps import StampStore

VERSION = "9.9"
TAGS = {"cherry": "CHR", "phagcn": "PGN", "phavip": "PHA_ORFs_PHV"}


@pytest.fixture
def ws(tmp_workspace):
    for tool, tag in TAGS.items():
        for vtool in ("geN", "VIB", "VS2"):
            matrix = extracted_matrix([(f"{vtool}|S1", 1, 0), (f"{vtool}|S2", 0, 1)], ["t1", "t2"])
            tmp_workspace.write_extracted(tool, f"{vtool}_{tag}_FEAT.csv", matrix)
            tmp_workspace.write_preprocessed(
                tool, f"{vtool}_{tag}_M_PP.csv", pd.DataFrame({"Accession": [f"{vtool}|S1_c1"]})
            )
    meta = tmp_workspace.data_root / "genomes" / "genome_metadata.csv"
    meta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["S1", "S2"], "label": [1, 0]}).to_csv(meta, sep=";", index=False)
    return tmp_workspace


@pytest.fixture
def ctx(ws) -> RunContext:
    return RunContext.load(ws.config_path)


def jobs_for(stage: str, ctx: RunContext, **params):
    return build_jobs(parse_run_doc({"steps": [{"stage": stage, **params}]}), ctx)


def validate_stage(stage: str, ctx: RunContext, **params):
    """Full validation, as ``phantom validate`` does it: build + data checks."""
    jobs = jobs_for(stage, ctx, **params)
    return plan(jobs, ctx, StampStore(ctx.section_path("stamps")), force=True)


def run_stage(stage: str, ctx: RunContext, **params) -> list[Path]:
    run = parse_run_doc({"steps": [{"stage": stage, **params}]})
    store = StampStore(ctx.section_path("stamps"))
    return execute(plan(build_jobs(run, ctx), ctx, store), ctx, store, run)


def fake_executor(monkeypatch, module, cls_name: str) -> list:
    """Replace an executor class whose run() creates <out_dir>/run_x and that
    exposes ``saver.exp_dir``; returns the list of configs it was built with."""
    seen: list = []

    class Saver:
        def __init__(self, exp_dir):
            self.exp_dir = exp_dir

    class Fake:
        def __init__(self, config):
            self.config = config
            self.saver = None

        def run(self):
            exp = Path(self.config.out_dir) / f"run_{len(seen)}"
            exp.mkdir(parents=True)
            self.saver = Saver(str(exp))
            seen.append(self.config)

    monkeypatch.setattr(module, cls_name, Fake)
    return seen
