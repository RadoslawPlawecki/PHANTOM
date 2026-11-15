"""
End to end: ``phantom example`` + ``phantom run runs/example.toml`` -- the
whole Python pipeline (raw folder -> genomes -> masks -> features ->
classifiers -> figures -> feature optimization) on the synthetic workspace,
through the command line exactly as a user runs it. The run happens once per
module; the tests check what a user relies on: every step's outputs, the
provenance records, that a rerun has nothing to do and that the same seed
gives the same results. Whether the *numbers* are unchanged is the
regression tests' job (tests/regression/).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from phantom.analysis.classification.machine_learning.models import MODELS
from phantom.example import N_CASES, N_CONTROLS
from phantom.runner.cli import main
from tests._helpers.example_run import EXAMPLE_RUN_FILE, example_workspace

RUNFILE = EXAMPLE_RUN_FILE
N_SAMPLES = N_CASES + N_CONTROLS


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    root = tmp_path_factory.mktemp("example")
    with example_workspace(root) as (config, exit_code):
        yield root, config, exit_code


def _run_dirs(root: Path, kind: str) -> list[Path]:
    return sorted(p for p in (root / "results" / kind / "1.0").iterdir() if p.is_dir())


def test_the_run_succeeds(example):
    _, _, code = example
    assert code == 0


def test_samples_get_ids_links_and_labels(example):
    root, _, _ = example
    metadata = pd.read_csv(root / "data" / "genomes" / "genome_metadata.csv", sep=";")
    assert sorted(metadata["id"], key=lambda s: int(s[1:])) == [f"S{n}" for n in range(1, 13)]
    assert metadata["label"].value_counts().to_dict() == {1: N_CASES, 0: N_CONTROLS}
    links = sorted((root / "data" / "assemblies").glob("S*.fa"))
    assert len(links) == N_SAMPLES and all(link.is_symlink() for link in links)
    # The CheckV counts come from the masks.
    assert {"checkv_vib_total", "checkv_vib_medium", "checkv_vib_high"} <= set(metadata.columns)
    assert (metadata["checkv_vib_medium"] <= metadata["checkv_vib_total"]).all()


def test_every_classification_run_is_there_with_its_seed(example):
    root, _, _ = example
    single = _run_dirs(root, "single-modal")
    multi = _run_dirs(root, "multi-modal")
    assert len(single) == 9 + 4  # 3 modalities x 3 tools (catboost) + 4 other models
    assert len(multi) == 6  # early + late fusion x 3 identification tools
    for run in single + multi:
        metadata = json.loads((run / "metadata.json").read_text())
        assert metadata["seed"] == 42
        assert metadata["labels"]["class_counts"] == {"0": N_CONTROLS, "1": N_CASES}
        assert (run / "loocv" / "metrics.json").is_file()
    assert all((run / "rcv" / "predictions.csv").is_file() for run in single)
    late = [r for r in multi if "_late_" in r.name]
    assert all(json.loads((r / "metadata.json").read_text())["opt"] for r in late)
    assert len(_run_dirs(root, "mds")) == 3


def test_figures_and_the_metrics_table(example):
    root, _, _ = example
    figures = root / "results" / "figures" / "1.0" / "catboost"
    for kind, validators in (("single", ("loocv", "rcv")), ("multi", ("loocv",))):
        for validator in validators:
            for name in ("metrics", "confusion_matrices", "misclassified", "top_features"):
                assert (figures / f"{name}_{kind}_{validator}.pdf").stat().st_size > 1000
    assert not (figures / "metrics_multi_rcv.pdf").exists()  # no run: no empty figure
    table = pd.read_csv(figures / "metrics.csv")
    assert set(table["kind"]) == {"single", "multi"}
    assert table["score"].between(-1, 1).all()


def test_the_feature_optimization_ran_every_strategy(example):
    root, _, _ = example
    output = root / "results" / "optimization" / "1.0" / "phagcn" / "geN"
    exploratory = json.loads((output / "exploratory" / "exploratory_result.json").read_text())
    assert set(exploratory["best_params"]) == {"binary", "min_patients"}
    permutation = json.loads((output / "permutation" / "permutation_test.json").read_text())
    assert len(permutation["null_scores"]) == 2 and 0 < permutation["p_value"] <= 1
    assert (output / "nested" / "full" / "metrics.json").is_file()
    assert json.loads((output / "metadata.json").read_text())["seed"] == 42


def test_outputs_record_their_provenance(example):
    root, _, _ = example
    run_toml = root / "results" / "figures" / "1.0" / "catboost" / "run.toml"
    record = tomllib.loads(run_toml.read_text())
    (step,) = record["steps"]
    assert step["stage"] == "classification.figures" and step["model"] == "catboost"
    assert record["provenance"]["inputs"] == ["results/single-modal/1.0", "results/multi-modal/1.0"]
    assert record["provenance"]["git_commit"]


def test_a_rerun_has_nothing_to_do(example, capsys):
    _, config, _ = example
    capsys.readouterr()
    assert main(["--config", str(config), "plan", str(RUNFILE)]) == 0
    plan = capsys.readouterr().out
    assert " RUN " not in plan
    assert "up to date" in plan


def test_the_same_seed_gives_the_same_result(example):
    """The same job again (into another folder, so the other tests still see
    the original run set) gives byte-identical results."""
    root, config, _ = example
    args = [
        "--set",
        "modality=comp",
        "--set",
        "vtool=geN",
        "--set",
        "classification.single.out_dir=results/repeat",
        "--only",
        "classification.single",
    ]
    assert main(["--config", str(config), "run", str(RUNFILE), *args]) == 0
    repeat = root / "results" / "repeat"
    for model in MODELS:  # both classification.single steps ran again: every model
        (new,) = [r for r in repeat.iterdir() if r.name.startswith(f"run_{model}_geN_comp_")]
        (old,) = [
            r
            for r in _run_dirs(root, "single-modal")
            if r.name.startswith(f"run_{model}_geN_comp_")
        ]
        for validator in ("loocv", "rcv"):
            for name in ("metrics.json", "predictions.csv", "feature_importance.csv"):
                assert (new / validator / name).read_text() == (old / validator / name).read_text()


def test_every_algorithm_ran_and_has_figures(example):
    root, _, _ = example
    for model in MODELS:
        (run,) = [
            r
            for r in _run_dirs(root, "single-modal")
            if r.name.startswith(f"run_{model}_geN_comp_")
        ]
        for validator in ("loocv", "rcv"):
            assert (run / validator / "metrics.json").is_file()
        assert (root / "results" / "figures" / "1.0" / model / "metrics_single_loocv.pdf").is_file()
    hgb = next(r for r in _run_dirs(root, "single-modal") if r.name.startswith("run_hgb_"))
    importance = pd.read_csv(hgb / "loocv" / "feature_importance.csv")["importance"]
    assert importance.isna().all()  # no built-in importances, reported as such
    assert not pd.read_csv(hgb / "rcv" / "feature_importance.csv")["importance"].isna().any()
