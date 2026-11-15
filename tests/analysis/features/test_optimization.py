"""Feature optimization: scoring one configuration, the Optuna study, and the
nested-CV and permutation strategies on a small phagcn table with a signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phantom.analysis.classification.machine_learning.cross_validation import (
    LeaveOneOutCrossValidator,
)
from phantom.analysis.classification.machine_learning.models import SingleModalModel
from phantom.analysis.features.optimization import nested_cv, permutation_test, search
from phantom.analysis.features.optimization.scoring import FeatureConfigScorer
from tests._helpers.fakes import FakeEstimator

N = 12  # patients S1..S12; S1..S6 are cases


@pytest.fixture
def labeled(tmp_workspace):
    """The workspace's metadata labels S1..S6 as cases, S7..S12 as controls."""
    metadata = tmp_workspace.data_root / "genomes" / "genome_metadata.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    ids = [f"S{n}" for n in range(1, N + 1)]
    pd.DataFrame({"id": ids, "label": [1] * 6 + [0] * 6}).to_csv(metadata, sep=";", index=False)
    return tmp_workspace


def _phagcn_table() -> pd.DataFrame:
    """Merged phagcn rows: cases carry genus A, controls genus B, everyone C."""
    rows = []
    for n in range(1, N + 1):
        genus = "A" if n <= 6 else "B"
        rows += [(f"geN|S{n}_k{n}_1", genus), (f"geN|S{n}_k{n}_2", "C")]
    return pd.DataFrame(rows, columns=["Accession", "genus"])


def _fake_model(**_):
    return SingleModalModel(FakeEstimator(threshold=0.5))


@pytest.fixture
def fast_models(monkeypatch):
    monkeypatch.setitem(
        search.MODEL_FACTORIES, "catboost", lambda use_smote=False, seed=42: _fake_model()
    )


# scoring


def _scorer():
    return FeatureConfigScorer(
        model=_fake_model(), validator=LeaveOneOutCrossValidator(verbose=False), min_features=1
    )


def test_a_configuration_with_a_signal_scores_high(labeled):
    matrix = search.apply_phagcn_config(_phagcn_table(), binary=True, min_patients=1)
    assert _scorer().run(matrix) == pytest.approx(1.0)


def test_shuffled_labels_override_the_metadata(labeled):
    matrix = search.apply_phagcn_config(_phagcn_table(), binary=True, min_patients=1)
    flipped = pd.Series({f"geN|S{n}": int(n > 6) for n in range(1, N + 1)})
    assert _scorer().run(matrix, y_override=flipped) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="does not cover"):
        _scorer().run(matrix, y_override=flipped.iloc[:3])


def test_unusable_configurations_score_zero(labeled):
    empty = pd.DataFrame(columns=["id", "A"])
    assert _scorer().run(empty) == 0.0
    too_few = search.apply_phagcn_config(_phagcn_table(), binary=True, min_patients=1)[["id", "A"]]
    assert (
        FeatureConfigScorer(
            model=_fake_model(), validator=LeaveOneOutCrossValidator(verbose=False)
        ).run(too_few)
        == 0.0
    )


def test_a_model_that_cannot_learn_scores_zero(labeled, monkeypatch):
    from catboost import CatBoostError

    class Failing:
        def run(self, *args):
            raise CatBoostError("All features are either constant or ignored.")

    matrix = search.apply_phagcn_config(_phagcn_table(), binary=True, min_patients=1)
    scorer = FeatureConfigScorer(model=_fake_model(), validator=Failing(), min_features=1)
    assert scorer.run(matrix) == 0.0


# the Optuna study


def test_the_phagcn_search_is_reproducible_with_a_seed(labeled, fast_models):
    run = lambda seed: search.search_phagcn(  # noqa: E731
        _phagcn_table(), "catboost", "loocv", "mcc", False, n_trials=4, verbose=False, seed=seed
    )
    a, b = run(1), run(1)
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
    assert a.best_value == pytest.approx(1.0)


# nested CV


def test_nested_cv_never_tests_a_patient_it_trained_on(labeled, fast_models, monkeypatch):
    seen_train = []
    real_search = search.SEARCH_FNS["phagcn"]

    def recording_search(df, *args, **kwargs):
        seen_train.append(set(df["Accession"].str.split("_").str[0]))
        return real_search(df, *args, **{**kwargs, "n_trials": 2})

    monkeypatch.setitem(nested_cv.SEARCH_FNS, "phagcn", recording_search)
    result = nested_cv.run_nested_cv(
        "phagcn",
        _phagcn_table(),
        "catboost",
        "loocv",
        "mcc",
        False,
        outer_folds=3,
        outer_repeats=2,
        top_k=1,
        random_state=3,
    )
    fold_keys = list(zip(result.full.repeat, result.full.fold, strict=True))
    for i, (repeat, fold) in enumerate(sorted(set(fold_keys))):
        tested = {
            pid
            for pid, key in zip(result.full.patient_ids, fold_keys, strict=True)
            if key == (repeat, fold)
        }
        assert tested.isdisjoint(seen_train[i])
    # Every patient is tested once per repeat.
    assert sorted(result.full.patient_ids) == sorted([f"geN|S{n}" for n in range(1, N + 1)] * 2)
    assert len(result.per_fold_top_features) == 6
    assert result.ablation.y_true.shape == result.full.y_true.shape


def test_nested_cv_is_reproducible_with_a_seed(labeled, fast_models):
    run = lambda: nested_cv.run_nested_cv(  # noqa: E731
        "phagcn",
        _phagcn_table(),
        "catboost",
        "loocv",
        "mcc",
        False,
        outer_folds=3,
        outer_repeats=1,
        inner_trials=2,
        top_k=1,
        random_state=5,
    )
    a, b = run(), run()
    assert np.array_equal(a.full.patient_ids, b.full.patient_ids)
    assert a.per_fold_best_params == b.per_fold_best_params


# permutation test


def test_the_permutation_null_is_reproducible_and_beaten_by_the_real_signal(labeled, fast_models):
    run = lambda: permutation_test.run_permutation_test(  # noqa: E731
        "phagcn",
        _phagcn_table(),
        "catboost",
        "loocv",
        "mcc",
        False,
        n_permutations=4,
        permutation_trials=2,
        random_state=11,
    )
    a, b = run(), run()
    assert np.array_equal(a.null_scores, b.null_scores)
    assert a.observed_score == pytest.approx(1.0)
    assert a.p_value == pytest.approx(1 / 5)  # (1 + #null >= observed) / (1 + n)
    assert len(a.per_permutation_best_params) == 4


# the other searches and their inputs


def _cherry_table() -> pd.DataFrame:
    lineage = "d__Bacteria;p__P;c__C;o__O;f__F;g__{g};s__{g} sp"
    rows = []
    for n in range(1, N + 1):
        genus = "Escherichia" if n <= 6 else "Bacteroides"
        rows.append((f"geN|S{n}_k1", 1.0, lineage.format(g=genus), "Not found"))
        rows.append((f"geN|S{n}_k2", 1.0, lineage.format(g="Shared"), "Not found"))
    df = pd.DataFrame(
        rows, columns=["Accession", "CHERRYScore", "Host_NCBI_lineage", "Host_GTDB_lineage"]
    )
    from phantom.analysis.features.phabox2.cherry import CherryFeaturePipeline

    return CherryFeaturePipeline().preprocess(df)


def _phavip_table() -> pd.DataFrame:
    rows = []
    for n in range(1, N + 1):
        ann = "integrase" if n <= 6 else "major capsid protein"
        rows += [
            (f"geN|S{n}_k1:1-90(+)", 0.9, 80.0, ann),
            (f"geN|S{n}_k1:91-200(+)", 0.9, 80.0, "holin"),
        ]
    df = pd.DataFrame(rows, columns=["Accession", "coverage", "pident", "Annotation"])
    from phantom.analysis.features.phabox2.phavip import PhavipFeaturePipeline

    df["Category"] = PhavipFeaturePipeline().categorize_annotations(df["Annotation"])
    return df


def test_the_cherry_and_phavip_searches_find_the_signal(labeled, fast_models):
    cherry = search.search_cherry(
        _cherry_table(), "catboost", "loocv", "mcc", False, n_trials=6, verbose=False
    )
    assert set(cherry.best_params) == {"binary", "min_patients", "feature_col"}
    phavip = search.search_phavip(
        _phavip_table(), "catboost", "loocv", "mcc", False, n_trials=3, verbose=False
    )
    assert set(phavip.best_params) == {"min_coverage", "min_pident"}
    assert len(phavip.trials) == 3


def test_a_verbose_study_prints_its_result(labeled, fast_models, capsys, tmp_path):
    search.search_phagcn(
        _phagcn_table(),
        "catboost",
        "rcv",
        "mcc",
        False,
        n_trials=1,
        file_path=tmp_path / "x.csv",
    )
    output = capsys.readouterr().out
    assert "PHAGCN_FEATURE_OPTIMIZATION OPTIMIZATION FINISHED" in output and "exploratory" in output


def test_load_masked_preprocesses_each_tool(tmp_path):
    merged = tmp_path / "geN_PGN_M.csv"
    pd.DataFrame(
        {
            "Accession": ["S1_k1"],
            "Prokaryotic virus (Bacteriophages and Archaeal virus)": ["Y"],
            "GenusCluster": ["known_genus"],
            "Lineage": ["genus:Tequatrovirus"],
            "PhaGCNScore": ["0.99"],
        }
    ).to_csv(merged, sep=";", index=False)
    assert search.load_masked("phagcn", merged, None)["genus"].tolist() == ["Tequatrovirus"]
    with pytest.raises(ValueError, match="Unknown tool"):
        search.load_masked("other", merged, None)
