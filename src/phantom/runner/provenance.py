"""
Provenance: every job writes a ``run.toml`` into each of its output
directories -- a valid run file holding exactly that one job's resolved
parameters, so ``phantom run <output>/run.toml --force`` recomputes that
result. The ``[provenance]`` table is informational and ignored on load.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import tomlkit

from phantom.config.loader import ConfigLoader
from phantom.runner.stamps import params_dict, rel_to_workspace

RUN_TOML = "run.toml"


def phantom_version() -> str:
    try:
        return metadata.version("phantom")
    except metadata.PackageNotFoundError:
        return "unknown"


def git_state() -> dict[str, Any]:
    """Commit of the code that produced a result, and whether it had local edits."""
    try:

        def git(*args: str) -> str:
            return subprocess.run(  # noqa: S603 -- fixed argv, no shell
                ["git", "-C", str(ConfigLoader.PROJECT_ROOT), *args],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()

        return {
            "git_commit": git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return {}


def _toml_safe(value: Any) -> Any:
    """TOML has no null: drop None values (they mean "use the default")."""
    if isinstance(value, dict):
        return {k: _toml_safe(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_toml_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def render_run_toml(
    stage: str,
    params: Any,
    run_metadata: dict[str, Any],
    source: Path | None,
    inputs: list[Path],
) -> str:
    doc = tomlkit.document()
    doc.add(
        tomlkit.comment(
            "Written by phantom. Recompute this result: phantom run <this file> --force"
        )
    )
    run = tomlkit.table()
    for key in ("name", "description"):
        if run_metadata.get(key):
            run.add(key, run_metadata[key])
    if run:
        doc.add("run", run)
    prov = tomlkit.table()
    prov.add("source", str(source) if source else "")
    prov.add("created", datetime.now().isoformat(timespec="seconds"))
    prov.add("phantom_version", phantom_version())
    for key, value in git_state().items():
        prov.add(key, value)
    prov.add("inputs", [rel_to_workspace(p) for p in inputs])
    doc.add("provenance", prov)
    steps = tomlkit.aot()
    step = tomlkit.table()
    step.add("stage", stage)
    for key, value in _toml_safe(params_dict(params)).items():
        step.add(key, value)
    steps.append(step)
    doc.add("steps", steps)
    return tomlkit.dumps(doc)


def write_run_toml(out_dir: Path, **kwargs: Any) -> Path:
    path = Path(out_dir) / RUN_TOML
    path.write_text(render_run_toml(**kwargs), encoding="utf-8")
    return path
