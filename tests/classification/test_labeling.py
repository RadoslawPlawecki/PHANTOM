"""Labels come from the metadata CSV's ``label`` column, never from code."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from phantom.classification.data.labeling import Labeling
from phantom.preprocessing.metadata.mapping import MetadataManager

pytestmark = pytest.mark.critical


def _write_metadata(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)
    return path


@pytest.fixture
def labeling() -> Labeling:
    return Labeling(pd.Series({"S1": 1, "S2": 0, "S12": 1}))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("S12", "S12"),
        ("geN|S12", "S12"),
        ("geN|S12_k141_9", "S12"),
        ("VS2|S12_k141_9||full", "S12"),
        ("geN|S12_k141_9|provirus_2_9", "S12"),
    ],
)
def test_sample_id_normalizes_every_id_form(raw, expected):
    assert Labeling.sample_id(pd.Series([raw])).tolist() == [expected]


def test_derive_label_maps_from_label_table(labeling):
    ids = pd.Series(["geN|S1", "geN|S2_k1_5", "VIB|S12"])
    np.testing.assert_array_equal(labeling.derive_label(ids), [1, 0, 1])


def test_derive_label_does_not_confuse_prefix_numbers(labeling):
    # S1 must not match S12 (or vice versa).
    np.testing.assert_array_equal(labeling.derive_label(pd.Series(["geN|S12"])), [1])
    np.testing.assert_array_equal(labeling.derive_label(pd.Series(["geN|S2"])), [0])


def test_derive_label_raises_on_unlabeled_sample(labeling):
    with pytest.raises(ValueError, match="S99"):
        labeling.derive_label(pd.Series(["geN|S99"]))


@pytest.mark.parametrize("bad", [2, -1])
def test_rejects_non_binary_labels(bad):
    with pytest.raises(ValueError, match="Labels must be one of"):
        Labeling(pd.Series({"S1": bad}))


def test_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        Labeling(pd.Series([0, 1], index=["S1", "S1"]))


def test_from_metadata_reads_label_column(tmp_path):
    path = _write_metadata(
        tmp_path / "meta.csv",
        [{"id": "S1", "x": 5, "label": 1}, {"id": "S2", "x": 6, "label": 0}],
    )
    lab = Labeling.from_metadata(path)
    np.testing.assert_array_equal(lab.derive_label(pd.Series(["S2", "S1"])), [0, 1])


def test_from_metadata_skips_rows_without_label(tmp_path):
    path = _write_metadata(
        tmp_path / "meta.csv",
        [{"id": "S1", "label": 1}, {"id": "S2", "label": None}],
    )
    lab = Labeling.from_metadata(path)
    assert lab.labels.to_dict() == {"S1": 1}
    with pytest.raises(ValueError, match="S2"):
        lab.derive_label(pd.Series(["S2"]))


def test_from_metadata_without_label_column_raises(tmp_path):
    path = _write_metadata(tmp_path / "meta.csv", [{"id": "S1", "x": 1}])
    with pytest.raises(ValueError, match="No 'label' column"):
        Labeling.from_metadata(path)


def test_from_metadata_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Labeling.from_metadata(tmp_path / "nope.csv")


def test_provenance_hash_is_order_independent_and_content_sensitive():
    a = Labeling(pd.Series({"S1": 1, "S2": 0}))
    b = Labeling(pd.Series({"S2": 0, "S1": 1}))
    c = Labeling(pd.Series({"S1": 0, "S2": 0}))
    assert a.provenance()["sha256"] == b.provenance()["sha256"]
    assert a.provenance()["sha256"] != c.provenance()["sha256"]
    assert a.provenance()["class_counts"] == {0: 1, 1: 1}
    assert a.provenance()["n_samples"] == 2


def test_default_reads_configured_metadata_and_sees_relabeling(tmp_workspace):
    path = tmp_workspace.data_root / "genomes" / "genome_metadata.csv"  # fixture config
    _write_metadata(path, [{"id": "S1", "label": 1}])
    assert Labeling.default().labels.to_dict() == {"S1": 1}

    _write_metadata(path, [{"id": "S1", "label": 0}])
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert Labeling.default().labels.to_dict() == {"S1": 0}


def test_path_from_config_requires_metadata_path():
    with pytest.raises(ValueError, match="metadata_path"):
        MetadataManager.path_from_config({"genomes": {"path": "data/genomes"}})


class TestMetadataOverwrite:
    def test_collision_still_blocked_by_default(self, tmp_path):
        mm = MetadataManager(tmp_path / "m.csv")
        mm.save(pd.DataFrame({"id": ["S1"], "label": [1]}))
        with pytest.raises(ValueError, match="Duplicate"):
            mm.add_metadata_columns(pd.DataFrame({"id": ["S1"], "label": [0]}))

    def test_overwrite_replaces_only_listed_column(self, tmp_path):
        mm = MetadataManager(tmp_path / "m.csv")
        mm.save(pd.DataFrame({"id": ["S1", "S2"], "x": [5, 6], "label": [1, 1]}))
        mm.add_metadata_columns(
            pd.DataFrame({"id": ["S1", "S2"], "label": [0, 1]}), overwrite={"label"}
        )
        out = mm.load()
        assert out["label"].tolist() == [0, 1]
        assert out["x"].tolist() == [5, 6]

    def test_overwrite_does_not_unlock_other_columns(self, tmp_path):
        mm = MetadataManager(tmp_path / "m.csv")
        mm.save(pd.DataFrame({"id": ["S1"], "x": [5], "label": [1]}))
        with pytest.raises(ValueError, match="x"):
            mm.add_metadata_columns(
                pd.DataFrame({"id": ["S1"], "x": [9], "label": [0]}), overwrite={"label"}
            )


class TestExternalCSVLabeler:
    @staticmethod
    def _run(tmp_path, external_rows, **kwargs):
        from phantom.preprocessing.metadata.labeling import ExternalCSVLabeler

        csv = tmp_path / "ext.csv"
        pd.DataFrame(external_rows).to_csv(csv, index=False)
        meta = pd.DataFrame({"id": ["S1", "S2"], "geN_original_file": ["a.fna", "b.fna"]})
        return ExternalCSVLabeler(csv, **kwargs).assign_labels(meta)

    def test_assigns_labels_from_csv(self, tmp_path):
        rows = [{"id": "S1", "label": 1}, {"id": "S2", "label": 0}]
        assert self._run(tmp_path, rows) == {"S1": 1, "S2": 0}

    def test_sample_missing_from_csv_raises_instead_of_class_0(self, tmp_path):
        with pytest.raises(ValueError, match="S2"):
            self._run(tmp_path, [{"id": "S1", "label": 1}])

    def test_duplicate_join_key_raises(self, tmp_path):
        rows = [
            {"id": "S1", "label": 1},
            {"id": "S1", "label": 0},
            {"id": "S2", "label": 0},
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            self._run(tmp_path, rows)

    def test_non_binary_label_raises(self, tmp_path):
        rows = [{"id": "S1", "label": 1}, {"id": "S2", "label": 2}]
        with pytest.raises(ValueError, match="0 or 1"):
            self._run(tmp_path, rows)

    def test_missing_join_column_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Join column"):
            self._run(tmp_path, [{"sample": "S1", "label": 1}])

    def test_custom_join_key(self, tmp_path):
        rows = [
            {"geN_original_file": "a.fna", "label": 0},
            {"geN_original_file": "b.fna", "label": 1},
        ]
        assert self._run(tmp_path, rows, join_key="geN_original_file") == {"S1": 0, "S2": 1}

    def test_semicolon_separated_csv(self, tmp_path):
        from phantom.preprocessing.metadata.labeling import ExternalCSVLabeler

        csv = tmp_path / "ext.csv"
        csv.write_text("id;label\nS1;1\nS2;0\n", encoding="utf-8")
        meta = pd.DataFrame({"id": ["S1", "S2"]})
        assert ExternalCSVLabeler(csv).assign_labels(meta) == {"S1": 1, "S2": 0}
