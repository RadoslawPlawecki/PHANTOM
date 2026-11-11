"""tools.* stages: what is handed to Snakemake (a recording fake stands in
for it), dry runs, and that the Snakefile parses with the generated config."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from tests.stages.conftest import jobs_for, run_stage, validate_stage

import phantom.stages  # noqa: F401 -- registers the stages
from phantom.runner.loader import RunFileError
from phantom.runner.stamps import StampStore

FAKE = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    argv = sys.argv[1:]
    cfg = json.loads(Path(argv[argv.index("--configfile") + 1]).read_text())
    Path(sys.argv[0]).with_name("calls.jsonl").open("a").write(
        json.dumps({"argv": argv, "config": cfg}) + "\\n"
    )
    """
)

TOOLS_CONFIG = """
[phabox2]
path = "data/phabox2"

[phanotate]
path = "data/phanotate/orfs"
genes = "data/phanotate/genes"
bed = "data/phanotate/bed"
proteins = "data/phanotate/proteins"

[workflow]
snakefile = "workflow/Snakefile"
snakemake = ["{python}", "{fake}"]
workdir = ".phantom/snakemake"
cores = 4

[databases]
checkv = "db/checkv"
phabox2 = "db/phabox2"

[envs]
checkv = "base"
phabox2 = "phabox2"

[threads]
checkv = 2
phabox2 = 3
"""


@pytest.fixture
def tools_ws(ws):
    fake = ws.root / "fake_snakemake.py"
    fake.write_text(FAKE, encoding="utf-8")
    with open(ws.config_path, "a", encoding="utf-8") as f:
        f.write(TOOLS_CONFIG.format(python=sys.executable, fake=fake))
    for vtool in ("genomad", "vibrant"):
        d = ws.data_root / "genomes" / vtool
        d.mkdir(parents=True, exist_ok=True)
        (d / f"S1_{vtool[:3]}.fna").write_text(">c\nACGT\n", encoding="utf-8")
    return ws


@pytest.fixture
def tctx(tools_ws):
    from phantom.runner.stage import RunContext

    return RunContext.load(tools_ws.config_path)


def _calls(ws):
    path = ws.root / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_checkv_targets_and_config(tools_ws, tctx):
    run_stage("tools.checkv", tctx, vtools=["geN"], samples=["S1"], cores=8)
    (call,) = _calls(tools_ws)
    argv, cfg = call["argv"], call["config"]
    assert argv[-1] == "all_checkv"
    assert argv[argv.index("--cores") + 1] == "8"
    assert "--use-conda" in argv
    assert cfg["vtools"] == ["genomad"]  # abbreviation mapped to the tool name
    assert cfg["samples"] == ["S1"]
    assert cfg["threads"]["checkv"] == 2
    assert cfg["envs"]["checkv"] == "base"
    assert cfg["databases"]["checkv"] == str(tools_ws.root / "db" / "checkv")
    assert cfg["tags"] == {"cherry": "CHR", "phagcn": "PGN", "phavip": "PHV"}


def test_dry_run_is_not_stamped(tools_ws, tctx):
    run_stage("tools.checkv", tctx, vtools=["geN"], dry_run=True)
    (call,) = _calls(tools_ws)
    assert "--dry-run" in call["argv"]
    store = StampStore(tctx.section_path("stamps"))
    (job,) = jobs_for("tools.checkv", tctx, vtools=["geN"], dry_run=True)
    assert (
        store.status("tools.checkv", job.params, job.stage.inputs(job.params, tctx)).state == "run"
    )


def test_phabox2_runs_all_task_targets_in_one_call(tools_ws, tctx):
    run_stage("tools.phabox2", tctx, tasks=["cherry", "phagcn"], vtools=["geN"])
    (call,) = _calls(tools_ws)
    assert call["argv"][-2:] == ["all_phabox2_cherry", "all_phabox2_phagcn"]


def test_phavip_needs_phanotate_orfs(tools_ws, tctx):
    with pytest.raises(RunFileError, match=r"run tools\.phanotate first"):
        validate_stage("tools.phabox2", tctx, tasks=["phavip"], vtools=["geN"])


def test_missing_config_is_reported(ctx):
    # The plain fixture config has no [workflow]/[databases]/... sections.
    with pytest.raises(RunFileError) as exc:
        validate_stage("tools.checkv", ctx)
    text = "\n".join(exc.value.errors)
    assert "config.toml is missing [databases].checkv" in text
    assert "config.toml is missing [threads].checkv" in text


def test_unknown_vtool(tctx):
    with pytest.raises(RunFileError, match="vtool 'XYZ' not in \\[tools\\]"):
        jobs_for("tools.checkv", tctx, vtools=["XYZ"])


@pytest.mark.skipif(shutil.which("conda") is None, reason="needs conda + the snakemake env")
def test_snakefile_parses_with_generated_config(tools_ws, tctx, tmp_path):
    """Lists the Snakefile's rules with a config phantom generates."""
    from phantom.stages.tools import snakemake_config

    snakemake = Path(shutil.which("conda")).parents[1] / "envs" / "snakemake" / "bin" / "snakemake"
    if not snakemake.exists():
        pytest.skip("no 'snakemake' conda environment")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(snakemake_config(tctx, ["geN"], None)), encoding="utf-8")
    out = subprocess.run(  # noqa: S603
        [
            str(snakemake),
            "--snakefile",
            str(Path(__file__).resolve().parents[2] / "workflow" / "Snakefile"),
            "--configfile",
            str(cfg),
            "--directory",
            str(tmp_path),
            "--list-rules",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert {"all_checkv", "checkv", "phabox2_phavip", "setup_genomad"} <= set(out)


def test_envs_and_software_are_optional(tools_ws, tctx):
    """A fresh install has no [envs]/[software]: pinned envs/*.yaml are used."""
    text = tools_ws.config_path.read_text(encoding="utf-8")
    tools_ws.config_path.write_text(
        text.replace('[envs]\ncheckv = "base"\nphabox2 = "phabox2"\n', ""), encoding="utf-8"
    )
    from phantom.runner.stage import RunContext

    fresh = RunContext.load(tools_ws.config_path)
    assert "envs" not in fresh.config
    validate_stage("tools.checkv", fresh, vtools=["geN"])
    run_stage("tools.checkv", fresh, vtools=["geN"])
    (call,) = _calls(tools_ws)
    assert call["config"]["envs"] == {}
