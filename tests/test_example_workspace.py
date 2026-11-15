"""``phantom example``: the synthetic workspace the end-to-end test and new
users run the pipeline on."""

from __future__ import annotations

import pytest

from phantom.config.loader import ConfigLoader
from phantom.example import N_CASES, N_CONTROLS, write_example
from phantom.runner.cli import main


def test_the_example_is_a_complete_workspace(tmp_path):
    config = write_example(tmp_path / "ex")
    root = config.parent
    text = config.read_text(encoding="utf-8")
    assert text.startswith("# The example") and 'workspace = "."' in text
    raw = root / "data" / "raw"
    assert len(list((raw / "case").glob("*/final.contigs.fa"))) == N_CASES
    assert len(list((raw / "control").glob("*/final.contigs.fa"))) == N_CONTROLS
    for tool, abbreviation in (("genomad", "geN"), ("vibrant", "VIB"), ("virsorter2", "VS2")):
        assert (
            root / "data" / "checkv" / tool / f"S1_{abbreviation}_ChV" / "quality_summary.tsv"
        ).is_file()
        for task, tag in (("cherry", "CHR"), ("phagcn", "PGN"), ("phavip", "PHA_ORFs_PHV")):
            assert (
                root / "data" / "phabox2" / task / tool / f"S1_{abbreviation}_{tag}.tsv"
            ).is_file()
    # VirSorter2 suffixes its contig ids, and PHANOTATE/phavip keep them.
    orfs = (root / "data" / "phanotate" / "orfs" / "virsorter2" / "S1_VS2_PHA_ORFs.fna").read_text()
    assert "||full:" in orfs


def test_the_example_is_deterministic(tmp_path):
    a, b = write_example(tmp_path / "a").parent, write_example(tmp_path / "b").parent
    for rel in (
        "data/raw/case/patient_03/final.contigs.fa",
        "data/phabox2/cherry/genomad/S3_geN_CHR.tsv",
    ):
        assert (a / rel).read_text() == (b / rel).read_text()


def test_the_example_refuses_a_non_empty_folder(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError, match="not empty"):
        write_example(tmp_path / "x")


def test_the_example_command(tmp_path, capsys):
    assert main(["example", str(tmp_path / "ex")]) == 0
    output = capsys.readouterr().out
    assert "phantom --config" in output and "runs/example.toml" in output
    assert main(["example", str(tmp_path / "ex")]) == 1  # not empty any more
    assert "not empty" in capsys.readouterr().err
    assert ConfigLoader.ACTIVE_PATH is None  # writing an example activates nothing
