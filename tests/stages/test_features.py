"""Feature-building stages on real (tiny) data, including a full chain on a
version that the same run file creates first."""

from __future__ import annotations

import pandas as pd
import pytest

import phantom.stages  # noqa: F401 -- registers the stages
from phantom.runner.loader import RunFileError, build_jobs, parse_run_doc
from phantom.runner.planner import DONE, execute, plan
from phantom.runner.stamps import StampStore
from tests._helpers.frames import _BACTEROIDES, _BACTEROIDES_GTDB
from tests.stages.conftest import VERSION, validate_stage


def _raw_cherry(tmp_path):
    """<tmp>/src/cherry/genomad/S{1,2}_geN_CHR.tsv, shaped like CHERRY output."""
    src = tmp_path / "src" / "cherry" / "genomad"
    src.mkdir(parents=True)
    for sample, contigs in (("S1", ["k149_1", "k149_2"]), ("S2", ["k149_3"])):
        pd.DataFrame(
            {
                "Accession": contigs,
                "CHERRYScore": [1.0] * len(contigs),
                "Host_NCBI_lineage": [_BACTEROIDES] * len(contigs),
                "Host_GTDB_lineage": [_BACTEROIDES_GTDB] * len(contigs),
            }
        ).to_csv(src / f"{sample}_geN_CHR.tsv", sep="\t", index=False)
    return tmp_path / "src" / "cherry"


def _run(doc, context):
    run = parse_run_doc(doc)
    store = StampStore(context.section_path("stamps"))
    planned = plan(build_jobs(run, context), context, store)
    return run, store, planned


def test_full_chain_on_a_version_created_by_the_same_run(ws, context, tmp_path):
    source = _raw_cherry(tmp_path)
    doc = {
        "steps": [
            {
                "stage": "features.import",
                "version": "7.0",
                "create": True,
                "sources": [str(source)],
            },
            {"stage": "features.merge", "version": "7.0", "id_columns": {"*": "Accession"}},
            {"stage": "features.preprocess", "version": "7.0", "tools": ["cherry"]},
            {
                "stage": "features.extract",
                "version": "7.0",
                "tools": ["cherry"],
                "feature_col": {"cherry": "ncbi_genus"},
                "binary": False,
                "min_patients": 1,
            },
        ]
    }
    run, store, planned = _run(doc, context)
    # Everything after the import reads what an earlier job writes.
    assert [planned_job.deferred for planned_job in planned] == [False, True, True, True]
    execute(planned, context, store, run)

    assert "7.0" in context.feature_versions()
    merged = pd.read_csv(
        context.stage_dir("7.0", "raw_merged") / "cherry" / "geN_CHR_M.csv", sep=";"
    )
    assert sorted(merged["Accession"]) == ["S1_k149_1", "S1_k149_2", "S2_k149_3"]
    matrix = pd.read_csv(
        context.stage_dir("7.0", "extracted") / "cherry" / "geN_CHR_FEAT.csv", sep=";"
    )
    assert matrix.set_index("id")["Bacteroides"].to_dict() == {"geN|S1": 2, "geN|S2": 1}

    # Re-planning the same file: the import now fails validation (the
    # version exists), so plan the rest -- all of it is up to date.
    rest = {"steps": doc["steps"][1:]}
    _, _, planned = _run(rest, context)
    assert [planned_job.state for planned_job in planned] == [DONE, DONE, DONE]


def test_merge_reports_unmatched_directories_and_missing_columns(ws, context, tmp_path):
    raw = context.stage_dir(VERSION, "raw") / "cherry" / "genomad"
    raw.mkdir(parents=True)
    pd.DataFrame({"Accession": ["k1"], "x": [1]}).to_csv(
        raw / "S1_geN_CHR.tsv", sep="\t", index=False
    )
    with pytest.raises(RunFileError, match=r"no pattern matches raw/cherry/genomad"):
        validate_stage(
            "features.merge", context, version=VERSION, id_columns={"phavip/*": "Genome"}
        )
    with pytest.raises(RunFileError, match=r"'Genome' \(for raw/cherry/genomad\) is not a column"):
        validate_stage("features.merge", context, version=VERSION, id_columns={"*": "Genome"})


def test_extract_validates_options_against_tools(context):
    with pytest.raises(RunFileError) as exc:
        validate_stage(
            "features.extract",
            context,
            version=VERSION,
            feature_col={"phavip": "x"},
            binary={"phavip": True},
            min_patients=11,
        )
    text = "\n".join(exc.value.errors)
    assert "feature_col: 'phavip' is not a tool of this step that uses a column" in text
    assert "binary: 'phavip' is not a tool of this step that uses matrix options" in text
    assert "min_patients must be 1-10, got 11" in text


def test_extract_needs_a_real_feature_column(ws, context):
    with pytest.raises(RunFileError, match="feature_col: give a column for cherry"):
        validate_stage(
            "features.extract", context, version=VERSION, feature_col={"phagcn": "genus"}
        )
    with pytest.raises(RunFileError, match="'nope' is not a column of cherry/"):
        validate_stage(
            "features.extract",
            context,
            version=VERSION,
            feature_col={"cherry": "nope", "phagcn": "Accession"},
        )


def test_preprocess_reports_missing_mask(ws, context):
    merged = context.stage_dir(VERSION, "raw_merged") / "cherry"
    merged.mkdir(parents=True)
    pd.DataFrame({"Accession": ["S1_k1"]}).to_csv(merged / "geN_CHR_M.csv", sep=";", index=False)
    with pytest.raises(RunFileError, match=r"mask not found: .*genomad/MQ_vMAGs\.tsv"):
        validate_stage("features.preprocess", context, version=VERSION, mask="MQ_vMAGs.tsv")


def test_import_rejects_existing_version_and_missing_source(context, tmp_path):
    with pytest.raises(RunFileError, match="already exists"):
        validate_stage("features.import", context, version=VERSION, create=True, sources=["x"])
    with pytest.raises(RunFileError, match="source not found"):
        validate_stage(
            "features.import", context, version=VERSION, sources=[str(tmp_path / "nope")]
        )


def test_the_step_that_created_a_version_can_run_again(ws, context, tmp_path):
    source = _raw_cherry(tmp_path)
    doc = {
        "steps": [
            {"stage": "features.import", "version": "8.0", "create": True, "sources": [str(source)]}
        ]
    }
    run, store, planned = _run(doc, context)
    execute(planned, context, store, run)
    # Same step again: up to date, not "version already exists".
    _, _, planned = _run(doc, context)
    assert [planned_job.state for planned_job in planned] == [DONE]
    # Another import creating the same version is still refused.
    other = {"steps": [{**doc["steps"][0], "sources": [str(source), str(source)]}]}
    with pytest.raises(RunFileError, match="already exists"):
        _run(other, context)


def test_the_importer_copies_replaces_only_on_request_and_checks_versions(ws, context, tmp_path):
    from phantom.analysis.features.importer import FeatureImporter

    src_dir = tmp_path / "srcs" / "cherry"
    src_dir.mkdir(parents=True)
    (src_dir / "a.tsv").write_text("old")
    single = tmp_path / "srcs" / "note.txt"
    single.write_text("v1")
    importer = FeatureImporter(config_path=context.config_path)
    raw, failed = importer.import_sources([src_dir, single], "6.0", create=True)
    assert failed == [] and (raw / "cherry" / "a.tsv").read_text() == "old"
    (src_dir / "a.tsv").write_text("new")
    single.write_text("v2")
    importer.import_sources([src_dir, single], "6.0", create=False)
    assert (raw / "cherry" / "a.tsv").read_text() == "old"  # kept: no overwrite
    importer.import_sources([src_dir, single], "6.0", create=False, overwrite=True)
    assert (raw / "cherry" / "a.tsv").read_text() == "new" and (
        raw / "note.txt"
    ).read_text() == "v2"
    with pytest.raises(ValueError, match="already exists"):
        importer.import_sources([single], "6.0", create=True)
    with pytest.raises(ValueError, match="does not exist"):
        importer.import_sources([single], "5.5", create=False)
    _, failed = importer.import_sources([tmp_path / "missing"], "6.0", create=False)
    assert failed and "missing" in failed[0]
