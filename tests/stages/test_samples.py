"""preprocessing.samples: any assembly file names -> S1..Sn, links, labels."""

from __future__ import annotations

import os

import pandas as pd
import pytest
from tests.stages.conftest import run_stage, validate_stage

import phantom.stages  # noqa: F401 -- registers the stages
from phantom.runner.loader import RunFileError


@pytest.fixture
def sheet(ws):
    """Three assemblies with arbitrary names, two in per-sample folders."""
    raw = ws.data_root / "raw"
    files = [
        raw / "run7" / "patient_A" / "final.contigs.fa",
        raw / "run7" / "patient_B" / "final.contigs.fa",
        raw / "ctrl-03.fasta",
    ]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(">c\nACGT\n", encoding="utf-8")
    path = ws.data_root / "samples.csv"
    path.write_text(
        "assembly,label\n"
        f"{files[0]},1\n"
        "raw/run7/patient_B/final.contigs.fa,1\n"  # relative = next to the sheet
        "raw/ctrl-03.fasta,0\n",
        encoding="utf-8",
    )
    (ws.data_root / "genomes" / "genome_metadata.csv").unlink()  # a fresh workspace
    return path, files


def _meta(ws):
    return pd.read_csv(ws.data_root / "genomes" / "genome_metadata.csv", sep=";").set_index("id")


def test_samples_are_numbered_in_row_order_linked_and_labeled(ws, ctx, sheet):
    _, files = sheet
    run_stage("preprocessing.samples", ctx)
    links = ws.data_root / "assemblies"
    assert sorted(p.name for p in links.glob("S*.fa")) == ["S1.fa", "S2.fa", "S3.fa"]
    for sid, f in zip(("S1", "S2", "S3"), files, strict=True):
        assert (links / f"{sid}.fa").resolve() == f.resolve()
    meta = _meta(ws)
    assert meta["label"].to_dict() == {"S1": 1, "S2": 1, "S3": 0}
    assert meta.loc["S3", "assembly"] == str(files[2].resolve())
    # Running again with the same sheet changes nothing.
    run_stage("preprocessing.samples", ctx)
    assert _meta(ws)["label"].to_dict() == {"S1": 1, "S2": 1, "S3": 0}


def test_new_samples_are_appended_and_labels_can_be_corrected(ws, ctx, sheet):
    path, _ = sheet
    run_stage("preprocessing.samples", ctx)
    new = ws.data_root / "raw" / "late.fa"
    new.write_text(">c\nA\n", encoding="utf-8")
    text = path.read_text(encoding="utf-8").replace("raw/ctrl-03.fasta,0", "raw/ctrl-03.fasta,1")
    path.write_text(text + "raw/late.fa,0\n", encoding="utf-8")
    run_stage("preprocessing.samples", ctx)
    assert _meta(ws)["label"].to_dict() == {"S1": 1, "S2": 1, "S3": 1, "S4": 0}


def test_reordering_the_sheet_would_swap_samples_and_is_refused(ws, ctx, sheet):
    path, _ = sheet
    run_stage("preprocessing.samples", ctx)
    header, *rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([header, rows[1], rows[0], rows[2]]) + "\n", encoding="utf-8")
    with pytest.raises(RunFileError, match=r"S1 was .*patient_A.*now makes it .*patient_B"):
        validate_stage("preprocessing.samples", ctx)


def test_moved_files_are_accepted_on_request(ws, ctx, sheet):
    path, files = sheet
    run_stage("preprocessing.samples", ctx)
    moved = ws.data_root / "moved.fasta"
    files[2].rename(moved)
    path.write_text(
        path.read_text(encoding="utf-8").replace("raw/ctrl-03.fasta", "moved.fasta"),
        encoding="utf-8",
    )
    with pytest.raises(RunFileError, match="accept_changes"):
        validate_stage("preprocessing.samples", ctx)
    run_stage("preprocessing.samples", ctx, accept_changes=True)
    assert os.readlink(ws.data_root / "assemblies" / "S3.fa") == str(moved.resolve())


def test_removing_a_sample_is_refused(ws, ctx, sheet):
    path, _ = sheet
    run_stage("preprocessing.samples", ctx)
    path.write_text(
        "\n".join(path.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8"
    )
    with pytest.raises(RunFileError, match=r"S3 are in the metadata .* not in the sample sheet"):
        validate_stage("preprocessing.samples", ctx)


def test_explicit_ids_are_kept_and_blank_ones_numbered_on(ws, ctx, sheet):
    path, files = sheet
    path.write_text(
        f"assembly;label;id\n{files[0]};1;S12\n{files[1]};1;\n{files[2]};0;S40\n", encoding="utf-8"
    )
    run_stage("preprocessing.samples", ctx)
    assert _meta(ws)["label"].to_dict() == {"S12": 1, "S40": 0, "S41": 1}


def test_existing_samples_without_recorded_files_need_explicit_ids(ws, ctx, sheet):
    # A workspace from before sample sheets: ids and labels, no assembly column.
    pd.DataFrame({"id": ["S1", "S2", "S3"], "label": [1, 1, 0]}).to_csv(
        ws.data_root / "genomes" / "genome_metadata.csv", sep=";", index=False
    )
    with pytest.raises(RunFileError, match="give the sheet an id column"):
        validate_stage("preprocessing.samples", ctx)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("file,label\nx.fa,1\n", r"missing column\(s\) assembly"),
        ("assembly,label\nnope.fa,1\n", "assembly not found"),
        ("assembly,label\nraw/ctrl-03.fasta,yes\n", "label must be 0 .* or 1"),
        ("assembly,label\nraw/ctrl-03.fasta,1\nraw/ctrl-03.fasta,0\n", "already listed in row 2"),
        ("assembly,label\nraw/a.fa.gz,1\n", "compressed"),
        ("assembly,label,id\nraw/ctrl-03.fasta,1,P7\n", "id must look like S1"),
        ("assembly,label\n", "no samples"),
    ],
)
def test_sheet_problems_are_reported_before_running(ws, ctx, sheet, text, message):
    path, _ = sheet
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RunFileError, match=message):
        validate_stage("preprocessing.samples", ctx)


def test_a_missing_sheet_is_reported(ctx, ws):
    with pytest.raises(RunFileError, match=r"sample sheet not found: .*samples\.csv"):
        validate_stage("preprocessing.samples", ctx)
