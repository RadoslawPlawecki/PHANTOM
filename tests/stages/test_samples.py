"""preprocessing.samples: a raw folder per class, any file names -> S1..Sn, links, labels."""

from __future__ import annotations

import os
import shutil

import pandas as pd
import pytest

import phantom.stages  # noqa: F401 -- registers the stages
from phantom.runner.loader import RunFileError
from tests.stages.conftest import run_stage, validate_stage


def _write(path, text=">c\nACGT\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def raw(ws):
    """Three assemblies with arbitrary names, two in per-sample folders."""
    raw = ws.data_root / "raw"
    files = [
        _write(raw / "case" / "patient_A" / "final.contigs.fa"),
        _write(raw / "case" / "patient_B" / "final.contigs.fa"),
        _write(raw / "control" / "ctrl-03.fasta"),
    ]
    _write(raw / "case" / "patient_A" / "log", "assembler log, not an assembly")
    (ws.data_root / "genomes" / "genome_metadata.csv").unlink()  # a fresh workspace
    return raw, files


def _metadata_table(ws):
    return pd.read_csv(ws.data_root / "genomes" / "genome_metadata.csv", sep=";").set_index("id")


def test_samples_are_numbered_in_path_order_linked_labeled_and_mapped(ws, context, raw):
    _, files = raw
    run_stage("preprocessing.samples", context)
    links = ws.data_root / "assemblies"
    assert sorted(link.name for link in links.glob("S*.fa")) == ["S1.fa", "S2.fa", "S3.fa"]
    for sample_id, f in zip(("S1", "S2", "S3"), files, strict=True):
        assert (links / f"{sample_id}.fa").resolve() == f.resolve()
    metadata = _metadata_table(ws)
    assert metadata["label"].to_dict() == {"S1": 1, "S2": 1, "S3": 0}
    assert metadata.loc["S3", "assembly"] == "control/ctrl-03.fasta"
    mapping = ws.data_root / "samples.csv"
    assert mapping.read_text(encoding="utf-8").splitlines() == [
        "id,assembly,label",
        "S1,case/patient_A/final.contigs.fa,1",
        "S2,case/patient_B/final.contigs.fa,1",
        "S3,control/ctrl-03.fasta,0",
    ]
    # Running again changes nothing, not even the mapping's modification time.
    written = mapping.stat().st_mtime_ns
    run_stage("preprocessing.samples", context)
    assert _metadata_table(ws)["label"].to_dict() == {"S1": 1, "S2": 1, "S3": 0}
    assert mapping.stat().st_mtime_ns == written


def test_new_samples_get_the_next_ids_whatever_their_names(ws, context, raw):
    folder, _ = raw
    run_stage("preprocessing.samples", context)
    _write(folder / "case" / "aaa_first_by_name.fa")
    _write(folder / "control" / "patient10.fna")
    _write(folder / "control" / "patient9.fna")
    run_stage("preprocessing.samples", context)
    metadata = _metadata_table(ws)
    assert metadata.loc["S1", "assembly"] == "case/patient_A/final.contigs.fa"  # kept
    assert metadata["assembly"].to_dict() == {
        "S1": "case/patient_A/final.contigs.fa",
        "S2": "case/patient_B/final.contigs.fa",
        "S3": "control/ctrl-03.fasta",
        "S4": "case/aaa_first_by_name.fa",
        "S5": "control/patient9.fna",  # natural order: 9 before 10
        "S6": "control/patient10.fna",
    }


def test_moving_a_sample_to_the_other_class_folder_relabels_it(ws, context, raw, capsys):
    folder, files = raw
    run_stage("preprocessing.samples", context)
    files[2].rename(folder / "case" / "ctrl-03.fasta")
    run_stage("preprocessing.samples", context)
    assert _metadata_table(ws)["label"].to_dict() == {"S1": 1, "S2": 1, "S3": 1}
    assert os.readlink(ws.data_root / "assemblies" / "S3.fa") == str(
        (folder / "case" / "ctrl-03.fasta").resolve()
    )
    assert "S3 (ctrl-03.fasta) relabeled 0 -> 1" in capsys.readouterr().out


def test_class_folders_may_carry_the_class_names(ws, context, raw):
    folder, _ = raw
    (folder / "case").rename(folder / "Allergy")
    (folder / "control").rename(folder / "healthy")
    context.config["classes"] = {"control": "Healthy", "case": "Allergy"}
    run_stage("preprocessing.samples", context)
    assert _metadata_table(ws)["label"].to_dict() == {"S1": 1, "S2": 1, "S3": 0}


def test_removing_a_sample_is_refused(ws, context, raw):
    _, files = raw
    run_stage("preprocessing.samples", context)
    files[2].unlink()
    with pytest.raises(RunFileError, match=r"S3 \(ctrl-03.fasta\) not found in the raw folder"):
        validate_stage("preprocessing.samples", context)


def test_files_named_like_ids_keep_them(ws, context, raw):
    folder, _ = raw
    shutil.rmtree(folder)
    _write(folder / "case" / "S12.fa")
    _write(folder / "control" / "S40" / "final.contigs.fa")
    _write(folder / "control" / "other.fa")
    run_stage("preprocessing.samples", context)
    assert _metadata_table(ws)["label"].to_dict() == {"S12": 1, "S40": 0, "S41": 0}


def test_existing_samples_without_recorded_files_need_their_ids_as_names(ws, context, raw):
    # A workspace from before the raw folder: ids and labels, no assembly column.
    pd.DataFrame({"id": ["S1", "S2", "S3"], "label": [1, 1, 0]}).to_csv(
        ws.data_root / "genomes" / "genome_metadata.csv", sep=";", index=False
    )
    with pytest.raises(RunFileError, match=r"S1, S2, S3 are in the metadata without .* S<n>"):
        validate_stage("preprocessing.samples", context)


def test_an_id_name_taken_by_another_sample_is_refused(ws, context, raw):
    folder, _ = raw
    run_stage("preprocessing.samples", context)
    _write(folder / "control" / "S1.fa")
    with pytest.raises(RunFileError, match=r"control/S1.fa is named S1, but S1 is patient_A"):
        validate_stage("preprocessing.samples", context)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda raw: _write(raw / "loose.fa"), "put it into a class folder"),
        (lambda raw: _write(raw / "sick" / "a.fa"), "unknown class folder 'sick'"),
        (lambda raw: _write(raw / "case" / "a.fa.gz"), "compressed"),
        (lambda raw: _write(raw / "case" / "patient_A" / "k21.contigs.fa"), "holds 2 FASTA"),
        (lambda raw: _write(raw / "case" / "empty" / "notes.txt"), "no FASTA file"),
        (lambda raw: _write(raw / "control" / "patient_B" / "final.contigs.fa"), "both classes"),
    ],
)
def test_raw_folder_problems_are_reported_before_running(context, raw, make, message):
    make(raw[0])
    with pytest.raises(RunFileError, match=message):
        validate_stage("preprocessing.samples", context)


def test_hidden_and_other_files_are_ignored(ws, context, raw):
    folder, _ = raw
    _write(folder / ".DS_Store", "")
    _write(folder / "README.txt", "my notes")
    _write(folder / "case" / ".hidden.fa")
    run_stage("preprocessing.samples", context)
    assert len(_metadata_table(ws)) == 3


def test_a_missing_or_empty_raw_folder_is_reported(context, ws):
    with pytest.raises(RunFileError, match=r"raw folder not found: .*raw .*case/ and control/"):
        validate_stage("preprocessing.samples", context)
    (ws.data_root / "raw" / "case").mkdir(parents=True)
    with pytest.raises(RunFileError, match="no assemblies in"):
        validate_stage("preprocessing.samples", context)


def test_class_names_that_collide_as_folders_are_refused(context, raw):
    context.config["classes"] = {"control": "Case", "case": "Sick"}
    with pytest.raises(RunFileError, match="names both classes"):
        validate_stage("preprocessing.samples", context)
