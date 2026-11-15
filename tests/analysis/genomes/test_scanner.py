"""Statistics scraped from the tool outputs: runtimes from each tool's logs,
CheckV quality counts (from the masks), assembly sizes."""

from __future__ import annotations

import pandas as pd
import pytest

from phantom.analysis.genomes.metadata import scanner
from phantom.config.loader import ConfigLoader


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def tool_outputs(tmp_workspace):
    data = tmp_workspace.data_root
    _write(
        data / "vibrant" / "S1_VIB" / "VIBRANT_S1" / "VIBRANT_log_run_S1.log",
        "VIBRANT v1.2.1\nRuntime:  98.3 minutes\n",
    )
    _write(data / "genomad" / "S1_geN" / "S1_annotate.log", "[09:00:00] start\n[09:10:00] more\n")
    _write(data / "genomad" / "S1_geN" / "S1_summary.log", "[09:20:00] x\n[09:42:00] finished\n")
    vs2 = data / "virsorter2" / "S1_VS2" / "log" / "iter-0"
    _write(
        vs2 / "step1-pp" / "circular-remove-partial-gene-common.log", "[2026-04-17 23:13 INFO] a\n"
    )
    _write(
        vs2 / "step2-extract-feature" / "extract-feature-from-hmmout-common.log",
        "[2026-04-18 01:44 INFO] b\n",
    )
    _write(data / "assemblies" / "S1.fa", ">c\nACGT\n")
    _write(data / "assemblies" / "notes.fa", ">c\nA\n")  # not a sample: ignored
    quality = "contig_id\tcompleteness\tcontamination\nk1\t90\t0\nk2\t55\t0\nk3\tNA\t0\n"
    _write(data / "checkv" / "vibrant" / "S1_VIB_ChV" / "quality_summary.tsv", quality)
    masks = data / "masks" / "vibrant"
    _write(masks / "MQ_provMAGs.tsv", "contig_id\nVIB|S1_k1\nVIB|S1_k2\n")
    _write(masks / "HQ_provMAGs.tsv", "contig_id\nVIB|S1_k1\n")
    return tmp_workspace


def test_runtimes_from_each_tools_logs(tool_outputs):
    data = tool_outputs.data_root
    assert scanner.runtime_vibrant(next((data / "vibrant").rglob("VIBRANT_log_run_*.log"))) == 98
    assert scanner.runtime_genomad(data / "genomad" / "S1_geN") == 42
    assert scanner.runtime_virsorter2(data / "virsorter2" / "S1_VS2") == 151  # crosses midnight
    assert scanner.runtime_virsorter2(data / "virsorter2") is None  # no logs


def test_sample_ids_and_sizes():
    assert scanner.get_sample_id("S12_VIB") == "S12"
    with pytest.raises(ValueError, match="sample_id"):
        scanner.get_sample_id("no-id-here")


def test_checkv_counts_come_from_the_masks(tool_outputs):
    data = tool_outputs.data_root
    summary = data / "checkv" / "vibrant" / "S1_VIB_ChV" / "quality_summary.tsv"
    assert scanner.count_checkv_scored(summary) == 2  # k3 has no estimate
    assert scanner.count_masked(data / "masks", "vibrant") == {
        "medium": {"S1": 2},
        "high": {"S1": 1},
    }
    with pytest.raises(FileNotFoundError, match=r"run preprocessing\.masks first"):
        scanner.count_masked(data / "masks", "genomad")


def test_a_file_that_is_not_a_quality_summary(tmp_path):
    path = _write(tmp_path / "x.tsv", "a\tb\n1\t2\n")
    assert scanner.count_checkv_scored(path) is None


def test_scan_metadata_joins_everything_on_the_sample_id(tool_outputs):
    config = ConfigLoader(tool_outputs.config_path).load()
    df = scanner.scan_metadata(config).set_index("id")
    row = df.loc["S1"]
    assert row["vib_runtime"] == 98 and row["gen_runtime"] == 42 and row["vs2_runtime"] == 151
    assert row["checkv_vib_total"] == 2
    assert row["checkv_vib_medium"] == 2 and row["checkv_vib_high"] == 1
    assert row["raw_size_bytes"] == len(">c\nACGT\n")
    without = scanner.scan_metadata(config, checkv=False)
    assert "checkv_vib_total" not in without.columns


def test_missing_tool_folders_are_skipped(tmp_workspace, capsys):
    assert scanner.build_index(tmp_workspace.data_root / "absent") == {}
    assert scanner.build_assembly_index(tmp_workspace.data_root / "absent") == {}
    assert scanner.build_checkv_index(tmp_workspace.data_root / "absent") == {}
    assert "not found" in capsys.readouterr().out


def test_the_metadata_table_stays_sorted_by_sample_number(tmp_path):
    from phantom.analysis.genomes.metadata.table import MetadataTable

    table = MetadataTable(tmp_path / "meta.csv")
    table.add_metadata_columns(pd.DataFrame({"id": ["S10", "S2"], "a": [1, 2]}))
    table.add_metadata_columns(pd.DataFrame({"id": ["S1"], "b": [3]}))
    assert table.load()["id"].tolist() == ["S1", "S2", "S10"]
    with pytest.raises(ValueError, match="Duplicate columns"):
        table.add_metadata_columns(pd.DataFrame({"id": ["S1"], "a": [9]}))
    table.add_metadata_columns(pd.DataFrame({"id": ["S1"], "a": [9]}), overwrite={"a"})
    # Overwritten columns are replaced wholesale, not merged per sample.
    assert table.load().set_index("id")["a"].dropna().to_dict() == {"S1": 9}
