"""preprocessing.* stages on tiny real inputs."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.stages.conftest import jobs_for, run_stage, validate_stage

import phantom.stages.preprocessing  # noqa: F401 -- registers the stages
from phantom.runner.loader import RunFileError


def _meta_path(ws):
    return ws.data_root / "genomes" / "genome_metadata.csv"


# --- collect -------------------------------------------------------------------


def test_collect_copies_genomes_and_records_renaming(ws, ctx):
    for sample in ("S1", "S2"):
        d = ws.data_root / "genomad" / f"{sample}_geN"
        d.mkdir(parents=True)
        (d / f"001_{sample}_x_virus.fna").write_text(">c\nACGT\n", encoding="utf-8")
    _meta_path(ws).unlink()  # collection creates the metadata from scratch
    run_stage("preprocessing.collect", ctx)
    assert sorted(p.name for p in (ws.data_root / "genomes" / "genomad").glob("*.fna")) == [
        "S1_geN.fna",
        "S2_geN.fna",
    ]
    meta = pd.read_csv(_meta_path(ws), sep=";")
    assert meta.set_index("id")["geN_formatted_file"].to_dict() == {
        "S1": "S1_geN.fna",
        "S2": "S2_geN.fna",
    }
    # Collecting again replaces the renaming columns instead of failing.
    run_stage("preprocessing.collect", ctx, auto_format=True)


# --- metadata --------------------------------------------------------------------


def test_metadata_labels_by_threshold_and_refuses_to_relabel(ws, ctx):
    pd.DataFrame({"id": ["S1", "S2", "S40"]}).to_csv(_meta_path(ws), sep=";", index=False)
    spec = {"strategy": "threshold", "threshold": 34}
    run_stage("preprocessing.metadata", ctx, targets=["labeling"], labeling=spec)
    meta = pd.read_csv(_meta_path(ws), sep=";")
    assert meta.set_index("id")["label"].to_dict() == {"S1": 1, "S2": 1, "S40": 0}
    with pytest.raises(RunFileError, match="already has labels; set overwrite_labels"):
        validate_stage("preprocessing.metadata", ctx, targets=["labeling"], labeling=spec)
    run_stage(
        "preprocessing.metadata",
        ctx,
        targets=["labeling"],
        labeling={"strategy": "manual", "positives": ["S40"]},
        overwrite_labels=True,
    )
    meta = pd.read_csv(_meta_path(ws), sep=";")
    assert meta.set_index("id")["label"].to_dict() == {"S1": 0, "S2": 0, "S40": 1}


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"targets": ["labeling"]}, "give labeling = { strategy"),
        (
            {"targets": ["runtimes"], "labeling": {"strategy": "threshold", "threshold": 3}},
            "not in targets",
        ),
        ({"labeling": {"strategy": "vote"}}, "labeling.strategy must be one of"),
        ({"labeling": {"strategy": "threshold"}}, "needs 'threshold'"),
        ({"labeling": {"strategy": "manual", "positives": "S1"}}, "must be a list"),
        ({"labeling": {"strategy": "threshold", "threshold": 3, "x": 1}}, "unknown key 'x'"),
    ],
)
def test_metadata_validation(ctx, params, message):
    with pytest.raises(RunFileError, match=message):
        jobs_for("preprocessing.metadata", ctx, **params)


# --- masks -----------------------------------------------------------------------


def _quality(ws, tool, sample, abbr, rows):
    d = ws.data_root / "checkv" / tool / f"{sample}_{abbr}_ChV"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["contig_id", "provirus", "completeness", "contamination"]).to_csv(
        d / "quality_summary.tsv", sep="\t", index=False
    )


def test_masks_apply_configured_thresholds(ws, ctx):
    _quality(
        ws,
        "genomad",
        "S1",
        "geN",
        [
            ("k1", "No", 80.0, 0.0),
            ("k2", "No", 60.0, 0.0),
            ("k3", "Yes", 90.0, 0.0),
            ("k4", "No", 99.0, 20.0),
        ],
    )
    run_stage("preprocessing.masks", ctx)
    masks = ws.data_root / "masks" / "genomad"

    def ids(name):
        return sorted(pd.read_csv(masks / name, sep="\t")["contig_id"])

    assert ids("MQ_vMAGs.tsv") == ["geN|S1_k1", "geN|S1_k2"]
    assert ids("HQ_vMAGs.tsv") == ["geN|S1_k1"]
    assert ids("HQ_provMAGs.tsv") == ["geN|S1_k1", "geN|S1_k3"]
    run_stage("preprocessing.masks", ctx, medium_completeness=70.0)
    assert ids("MQ_vMAGs.tsv") == ["geN|S1_k1"]


def test_masks_validation(ctx):
    with pytest.raises(RunFileError, match="medium_completeness <= high_completeness"):
        jobs_for("preprocessing.masks", ctx, medium_completeness=80.0, high_completeness=60.0)
    with pytest.raises(RunFileError, match=r"run tools\.checkv first"):
        validate_stage("preprocessing.masks", ctx)
