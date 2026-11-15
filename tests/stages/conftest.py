"""A workspace with extracted matrices for all three modalities x vtools,
preprocessed phagcn/cherry/phavip tables and a labeled metadata CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from phantom.runner.loader import build_jobs, parse_run_doc
from phantom.runner.planner import execute, plan
from phantom.runner.stage import RunContext
from phantom.runner.stamps import StampStore
from tests._helpers.frames import extracted_matrix

VERSION = "9.9"
TAGS = {"cherry": "CHR", "phagcn": "PGN", "phavip": "PHA_ORFs_PHV"}


@pytest.fixture
def ws(tmp_workspace):
    for tool, tag in TAGS.items():
        for virus_identification_tool in ("geN", "VIB", "VS2"):
            matrix = extracted_matrix(
                [
                    (f"{virus_identification_tool}|S1", 1, 0),
                    (f"{virus_identification_tool}|S2", 0, 1),
                ],
                ["t1", "t2"],
            )
            tmp_workspace.write_extracted(
                tool, f"{virus_identification_tool}_{tag}_FEAT.csv", matrix
            )
            tmp_workspace.write_preprocessed(
                tool,
                f"{virus_identification_tool}_{tag}_M_PP.csv",
                pd.DataFrame({"Accession": [f"{virus_identification_tool}|S1_c1"]}),
            )
    metadata = tmp_workspace.data_root / "genomes" / "genome_metadata.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["S1", "S2"], "label": [1, 0]}).to_csv(metadata, sep=";", index=False)
    return tmp_workspace


@pytest.fixture
def context(ws) -> RunContext:
    return RunContext.load(ws.config_path)


def jobs_for(stage: str, context: RunContext, **params):
    return build_jobs(parse_run_doc({"steps": [{"stage": stage, **params}]}), context)


def validate_stage(stage: str, context: RunContext, **params):
    """Full validation, as ``phantom validate`` does it: build + data checks."""
    jobs = jobs_for(stage, context, **params)
    return plan(jobs, context, StampStore(context.section_path("stamps")), force=True)


def run_stage(stage: str, context: RunContext, **params) -> list[Path]:
    run = parse_run_doc({"steps": [{"stage": stage, **params}]})
    store = StampStore(context.section_path("stamps"))
    return execute(plan(build_jobs(run, context), context, store), context, store, run)


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
