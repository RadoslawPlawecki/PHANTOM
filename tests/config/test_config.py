"""The configuration: merging config.local.toml, activating a --config for
the whole process (and its workspace), and the schema check."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from phantom.config import plotting
from phantom.config.loader import VIRUS_IDENTIFICATION_TOOLS_SECTION as VTOOLS
from phantom.config.loader import ConfigLoader, merge
from phantom.config.schema import ConfigError, check_config

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.toml"


def _repo_config() -> dict:
    with open(REPO_CONFIG, "rb") as f:
        return tomllib.load(f)


def test_local_settings_merge_key_by_key():
    base = {"workflow": {"cores": "all", "snakemake": ["snakemake"]}, "x": {"geN": "genomad"}}
    local = {"workflow": {"cores": 40}, "envs": {"genomad": "g"}}
    merged = merge(base, local)
    assert merged["workflow"] == {"cores": 40, "snakemake": ["snakemake"]}
    assert merged["envs"] == {"genomad": "g"} and merged["x"] == {"geN": "genomad"}


def test_a_missing_config_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="--config"):
        ConfigLoader(tmp_path / "nope.toml").load()


def test_a_missing_location_is_an_error_naming_the_key():
    with pytest.raises(ValueError, match=r"\[results\]\.path"):
        ConfigLoader.config_path({}, "results")


def test_activating_a_config_makes_it_the_process_default(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    ConfigLoader.activate(config)
    assert ConfigLoader().path == config.resolve()
    assert tmp_path != ConfigLoader.DATA_ROOT  # no workspace key: unchanged


def test_a_workspace_key_moves_the_data_workspace(tmp_path):
    config = tmp_path / "ws" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        'workspace = "."\n' + REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    ConfigLoader.activate(config)
    assert config.parent.resolve() == ConfigLoader.DATA_ROOT
    assert ConfigLoader.resolve_data_path("data/x") == config.parent.resolve() / "data" / "x"


def test_activating_an_invalid_config_lists_every_problem(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("workflw = {}\n[threads]\ncheckv = 0\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        ConfigLoader.activate(config)
    assert len(exc.value.errors) == 2
    assert "did you mean 'workflow'" in exc.value.errors[0]


# schema


def test_the_shipped_config_is_valid():
    assert check_config(_repo_config()) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.update(workspace=3), "workspace must be a folder"),
        (lambda c: c["genomad"].update(viral_fn="x"), "did you mean 'viral_fna'"),
        (lambda c: c["genomad"].update(path=""), r"\[genomad\]\.path must be a non-empty string"),
        (lambda c: c[VTOOLS].update(XX="megahit"), "not a virus identification tool"),
        (
            lambda c: c[VTOOLS].update(XX=1),
            r"\[virus-identification-tools\]\.XX must be a tool name",
        ),
        (lambda c: c.update(tools={}), r"\[tools\] is now called \[virus-identification-tools\]"),
        (lambda c: c["modalities"].update(bad="x"), r"\[modalities\]\.bad must be a table"),
        (lambda c: c["modalities"]["comp"].update(tool="phagcm"), "tool must be one of"),
        (lambda c: c["modalities"]["comp"].pop("tag"), "missing tag"),
        (lambda c: c["modalities"]["comp"].update(label=""), "label must be a non-empty string"),
        (lambda c: c["workflow"].update(snakemake="snakemake"), "a command as a list"),
        (lambda c: c["workflow"].update(cores=0), 'cores must be "all" or a positive number'),
        (lambda c: c["workflow"].update(snakefile=""), r"\[workflow\]\.snakefile"),
        (lambda c: c["workflow"].update(cores2=1), "unknown key 'cores2'"),
        (
            lambda c: c["threads"].update(checkv="all"),
            r"\[threads\]\.checkv must be a positive number",
        ),
        (lambda c: c["databases"].update(checkv=1), r"\[databases\]\.checkv must be a string"),
        (lambda c: c.update(results=[]), r"\[results\] must be a table"),
        (lambda c: c["features"].update(v1_0="x"), "feature versions are tables"),
        (lambda c: c["features"].update(v1_0={"path": "a"}), r"v1_0\]\.version must be"),
        (
            lambda c: c["features"].update(v1_0={"path": "a", "version": "1", "raw": {}}),
            r"raw must be a table with a path",
        ),
        (
            lambda c: c["features"].update(
                v1_0={"path": "a", "version": "1"}, v1_1={"path": "b", "version": "1"}
            ),
            "declared twice",
        ),
        (lambda c: c.update(sampels={}), "did you mean 'samples'"),  # codespell:ignore
    ],
)
def test_schema_problems_name_the_field(change, message):
    config = _repo_config()
    change(config)
    errors = check_config(config)
    assert any(__import__("re").search(message, e) for e in errors), errors


def test_plot_style_with_and_without_latex(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plotting, "latex_available", lambda: True)
    plotting.use_latex()
    assert plt.rcParams["text.usetex"] is True
    monkeypatch.setattr(plotting, "latex_available", lambda: False)
    plotting.use_latex()
    assert plt.rcParams["text.usetex"] is False


def test_class_names_come_from_the_config():
    from phantom.config.classes import class_names, slug

    assert class_names({}) == {0: "Control", 1: "Case"}
    assert class_names({"classes": {"control": "Healthy", "case": "Allergy"}}) == {
        0: "Healthy",
        1: "Allergy",
    }
    assert slug("Crohn's disease") == "crohn_s_disease"
    config = _repo_config()
    config["classes"]["case"] = " "
    config["classes"]["cases"] = "x"
    errors = check_config(config)
    assert any("[classes].case must be a non-empty name" in e for e in errors)
    assert any("did you mean 'case'" in e for e in errors)
