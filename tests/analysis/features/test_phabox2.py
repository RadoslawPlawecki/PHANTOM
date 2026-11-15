"""The per-task PhaBOX2 pipelines (cleaning + feature matrix) and the shared
matrix, loading and masking helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from phantom.analysis.features.phabox2.cherry import CherryFeaturePipeline
from phantom.analysis.features.phabox2.loading import (
    apply_mask,
    format_accession,
    load_file,
    split_taxonomy,
)
from phantom.analysis.features.phabox2.matrix import (
    build_matrix,
    build_taxonomy_matrix,
    derive_patient_id,
    filter_accessions,
)
from phantom.analysis.features.phabox2.phagcn import PhagcnFeaturePipeline
from phantom.analysis.features.phabox2.phatyp import PhatypFeaturePipeline
from phantom.analysis.features.phabox2.phavip import PhavipFeaturePipeline
from tests._helpers.frames import cherry_raw_df, phagcn_raw_df, phavip_raw_df

ESCHERICHIA = (
    "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;"
    "f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli"
)
NO_GENUS = "d__Bacteria;p__Bacillota;c__NAmissing;o__NAmissing;f__NAmissing;g__NAmissing;s__x"


# shared helpers


def test_patient_ids_and_well_formed_accessions():
    acc = pd.Series(["geN|S1_k141_1", "VS2|S12_k141_2||full"])
    assert derive_patient_id(acc).tolist() == ["geN|S1", "VS2|S12"]
    df = pd.DataFrame({"Accession": ["geN|S1_k1", "S1_k1", "geN|X_k1", None]})
    assert filter_accessions(df)["Accession"].tolist() == ["geN|S1_k1"]


def test_a_matrix_counts_or_flags_and_drops_rare_features():
    df = pd.DataFrame(
        {
            "Accession": ["geN|S1_a", "geN|S1_b", "geN|S2_c", "geN|S3_d"],
            "taxon": ["A", "A", "A", "B"],
        }
    )
    counts = build_matrix(df.copy(), "taxon", binary=False, min_patients=2)
    assert counts.set_index("id").to_dict() == {"A": {"geN|S1": 2, "geN|S2": 1, "geN|S3": 0}}
    flags = build_matrix(df.copy(), "taxon", binary=True, min_patients=1)
    assert flags.set_index("id")["A"].tolist() == [1, 1, 0]
    # A test fold is aligned to the train fold's columns, whatever its own counts.
    aligned = build_taxonomy_matrix(df.copy(), "taxon", min_patients=9, feature_columns=["B", "C"])
    assert list(aligned.columns) == ["id", "B", "C"]


def test_lineages_split_into_ranks():
    df = split_taxonomy(pd.DataFrame({"lin": [ESCHERICHIA]}), col="lin", lineage_type="ncbi")
    assert df.loc[0, "ncbi_genus"] == "Escherichia"
    plain = split_taxonomy(pd.DataFrame({"lin": [ESCHERICHIA]}), col="lin")
    assert plain.loc[0, "species"] == "Escherichia coli"


def test_merged_files_load_with_tool_prefixed_accessions(tmp_path):
    path = tmp_path / "VIB_CHR_M.csv"
    pd.DataFrame({"Accession": ["S1_k141_1"], "x": [1]}).to_csv(path, sep=";", index=False)
    assert load_file(path)["Accession"].tolist() == ["VIB|S1_k141_1"]
    assert format_accession("geN", pd.Series(["S1_k_1"])).tolist() == ["geN|S1_k1"]


def test_the_checkv_mask_keeps_listed_contigs_and_their_orfs(tmp_path):
    mask = tmp_path / "MQ_vMAGs.tsv"
    pd.DataFrame({"contig_id": ["VIB|S1_k141_1"]}).to_csv(mask, sep="\t", index=False)
    df = pd.DataFrame({"Accession": ["VIB|S1_k141_1", "VIB|S1_k141_1:1-90(+)", "VIB|S1_k141_2"]})
    assert apply_mask(df, mask)["Accession"].tolist() == ["VIB|S1_k141_1", "VIB|S1_k141_1:1-90(+)"]
    assert len(apply_mask(df, None)) == 3
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"id": ["x"]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="contig_id"):
        apply_mask(df, bad)


# cherry (host)


def test_cherry_keeps_confident_bacterial_hosts_with_a_genus():
    rows = [
        ("geN|S1_k1", 1.0, ESCHERICHIA, "Not found"),
        ("geN|S1_k2", 0.5, ESCHERICHIA, "Not found"),  # low score
        ("geN|S2_k3", "-", "-", "Not found"),  # no prediction
        ("geN|S2_k4", 1.0, NO_GENUS, "Not found"),  # no genus
    ]
    clean = CherryFeaturePipeline().preprocess(cherry_raw_df(rows))
    assert clean["Accession"].tolist() == ["geN|S1_k1"]
    assert clean.iloc[0]["ncbi_genus"] == "Escherichia"
    matrix = CherryFeaturePipeline().build_feature_matrix(
        clean, feature_col="ncbi_genus", binary=False, min_patients=1
    )
    assert matrix.set_index("id")["Escherichia"].to_dict() == {"geN|S1": 1}


# phagcn (taxonomy)


def test_phagcn_keeps_known_prokaryotic_genera_above_the_score():
    rows = [
        ("geN|S1_k1", "Y", "known_genus", "family:F;genus:Tequatrovirus", "0.99;0.95"),
        ("geN|S1_k2", "Y", "known_genus", "family:F;genus:Lowscore", "0.99;0.10"),
        ("geN|S2_k3", "N", "known_genus", "family:F;genus:Eukvirus", "1.0;1.0"),
        ("geN|S2_k4", "Y", "singleton", "family:F", "1.0"),
    ]
    clean = PhagcnFeaturePipeline().preprocess(phagcn_raw_df(rows))
    genus = clean.set_index("Accession")["genus"]
    assert genus["geN|S1_k1"] == "Tequatrovirus"
    assert pd.isna(genus["geN|S1_k2"])  # below the score: no genus
    assert len(genus) == 2
    matrix = PhagcnFeaturePipeline().build_feature_matrix(clean, min_patients=1)
    assert list(matrix.columns) == ["id", "Tequatrovirus"]


# phavip (function)


def test_phavip_categorizes_annotations_into_fixed_ratios():
    rows = [
        ("geN|S1_k1:1-90(+)", 0.9, 80.0, "major capsid protein"),
        ("geN|S1_k1:91-200(+)", 0.9, 80.0, "holin"),
        ("geN|S1_k1:201-300(+)", 0.9, 80.0, "something unheard of"),
        ("geN|S1_k1:301-400(+)", 0.1, 80.0, "integrase"),  # low coverage
        ("geN|S2_k2:1-90(+)", 0.9, 10.0, "integrase"),  # low identity
        ("geN|S2_k2:91-200(+)", 0.8, 50.0, "integrase"),
    ]
    pipeline = PhavipFeaturePipeline()
    clean = pipeline.preprocess(phavip_raw_df(rows))
    assert clean["Category"].tolist() == ["structural", "lysis", "other_function", "lysogeny"]
    matrix = pipeline.build_feature_matrix(clean).set_index("id")
    assert list(matrix.columns) == [f"{c}_ratio" for c in pipeline.ALL_CATEGORIES]
    assert matrix.loc["geN|S1", "structural_ratio"] == pytest.approx(0.3333)
    assert matrix.loc["geN|S2", "lysogeny_ratio"] == 1.0
    aligned = pipeline.build_feature_matrix(clean, feature_columns=["lysis_ratio"])
    assert list(aligned.columns) == ["id", "lysis_ratio"]


# phatyp (lifestyle)


def test_phatyp_lifestyle_ratios():
    df = pd.DataFrame(
        {
            "Accession": ["geN|S1_k1", "geN|S1_k2", "geN|S1_k3", "geN|S2_k4"],
            "TYPE": ["temperate", "virulent", "virulent", "virulent"],
            "PhaTYPScore": [0.95, 0.99, 0.5, "1.0"],
        }
    )
    pipeline = PhatypFeaturePipeline()
    clean = pipeline.preprocess(df)
    assert len(clean) == 3
    matrix = pipeline.build_feature_matrix(clean).set_index("id")
    assert matrix.loc["geN|S1"].to_dict() == {"temperate_ratio": 0.5, "virulent_ratio": 0.5}
    assert matrix.loc["geN|S2", "virulent_ratio"] == 1.0
