"""
Regression harness for the menu -> run-file migration.

Every pipeline action ("case", cases.py) runs as a ``phantom`` run-file step
in a throwaway sandbox workspace, and its outputs are reduced to a *manifest*
-- normalized relative path -> SHA-256 of the content. ``check`` requires a
run to reproduce the recorded baseline file for file; ``baseline`` records a
new one. The baselines in use were recorded from the former interactive menu
(answered identically), so passing them means the stages reproduce it. Outputs are deleted after hashing (the disk is small; some
cases write >1 GB), only small text outputs are kept next to the baseline for
diffing.

Safety: real inputs are symlinked into the sandbox read-only *by convention*,
so every linked tree is fingerprinted (path, size, mtime) before and after a
run and the harness aborts if anything under it changed.

Determinism: ``SOURCE_DATE_EPOCH`` pins PDF timestamps, every Optuna study
gets a seeded TPE sampler (the pipeline itself creates unseeded ones), and
the optimization trial budgets are reduced -- identically on both paths.
"""

from __future__ import annotations

import os

os.environ["SOURCE_DATE_EPOCH"] = "0"
os.environ["MPLBACKEND"] = "Agg"

import contextlib
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from phantom.config.loader import ConfigLoader

WORKSPACE = ConfigLoader.DATA_ROOT  # the real one, captured before any patching
REPO_CONFIG = ConfigLoader.DEFAULT_CONFIG_PATH
REG_ROOT = WORKSPACE / ".phantom" / "regression"
SANDBOX = REG_ROOT / "sandbox"
BASELINES = REG_ROOT / "baselines"
FAILED = REG_ROOT / "failed"

SKIP_NAMES = {"run.toml"}
SKIP_SUFFIXES = {".log"}
TIMESTAMP = re.compile(r"\d{8}_\d{6}")
KEEP_SUFFIXES = {".csv", ".tsv", ".json", ".toml"}
REDUCED_TRIALS = 3


@dataclass
class Case:
    name: str
    outputs: list[str]  # sandbox-relative paths whose content is the result
    steps: list[dict]  # run-file [[steps]] for the new path
    links: list[str] = field(default_factory=list)  # read-only real inputs
    copies: list[str] = field(default_factory=list)  # small real files the case may modify
    prepare: Callable[[Path], None] | None = None
    real: dict[str, str] = field(default_factory=dict)  # output -> real path, informational
    keep_limit_mb: int = 50


# --- sandbox -----------------------------------------------------------------


def build_sandbox(case: Case) -> None:
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir(parents=True)
    shutil.copy2(REPO_CONFIG, SANDBOX / "config.toml")
    local = REPO_CONFIG.with_name("config.local.toml")
    if local.is_file():  # this machine's settings and feature versions
        shutil.copy2(local, SANDBOX / "config.local.toml")
    for rel in case.links:
        src = WORKSPACE / rel
        if not src.exists():
            continue  # e.g. data/megahit is not present on every machine
        dst = SANDBOX / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=src.is_dir())
    for rel in case.copies:
        dst = SANDBOX / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE / rel, dst)
    if case.prepare:
        case.prepare(SANDBOX)


@contextlib.contextmanager
def sandboxed() -> Iterator[None]:
    saved = (ConfigLoader.DATA_ROOT, ConfigLoader.DEFAULT_CONFIG_PATH)
    ConfigLoader.DATA_ROOT = SANDBOX
    ConfigLoader.DEFAULT_CONFIG_PATH = SANDBOX / "config.toml"
    try:
        yield
    finally:
        ConfigLoader.DATA_ROOT, ConfigLoader.DEFAULT_CONFIG_PATH = saved


def _tree_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        st = path.stat()
        h.update(f"{st.st_size}|{st.st_mtime_ns}".encode())
        return h.hexdigest()
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            st = p.lstat()
            h.update(f"{p.relative_to(path)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


@contextlib.contextmanager
def guard_real_inputs(case: Case) -> Iterator[None]:
    linked = [WORKSPACE / rel for rel in case.links if (WORKSPACE / rel).exists()]
    before = {p: _tree_fingerprint(p) for p in linked}
    yield
    changed = [str(p) for p in linked if _tree_fingerprint(p) != before[p]]
    if changed:
        raise SystemExit(f"[ABORT] real input data changed during the run: {changed}")


# --- determinism patches -----------------------------------------------------


class Patcher:
    """Minimal monkeypatch: setattr now, restore on close."""

    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, Any]] = []

    def set(self, obj: Any, name: str, value: Any) -> None:
        self._undo.append((obj, name, obj.__dict__.get(name, getattr(obj, name, None))))
        setattr(obj, name, value)

    def close(self) -> None:
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)
        self._undo.clear()


@contextlib.contextmanager
def deterministic(p: Patcher) -> Iterator[None]:
    import optuna

    from phantom.analysis.features.optimization import search_core

    real_create = optuna.create_study

    def seeded_create_study(*args, **kwargs):
        if kwargs.get("sampler") is None:
            kwargs["sampler"] = optuna.samplers.TPESampler(seed=0)
        return real_create(*args, **kwargs)

    p.set(optuna, "create_study", seeded_create_study)

    # The MDS evaluation's PERMANOVA is called without a seed, so its p-value
    # varies from run to run.
    from phantom.analysis.classification.ml import evaluator

    real_permanova = evaluator.permanova
    p.set(evaluator, "permanova", lambda *a, **k: real_permanova(*a, **{"seed": 0, **k}))
    for name in ("CHERRY_DEFAULT_TRIALS", "PHAGCN_DEFAULT_TRIALS", "PHAVIP_DEFAULT_TRIALS"):
        p.set(search_core, name, REDUCED_TRIALS)
    p.set(search_core, "DEFAULT_TRIALS", dict.fromkeys(search_core.DEFAULT_TRIALS, REDUCED_TRIALS))
    yield


# --- manifests ---------------------------------------------------------------


def _iter_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    for dirpath, dirnames, filenames in os.walk(path):  # does not follow symlinks
        dirnames.sort()
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pdf_render_sha256(path: Path) -> str:
    """
    PDFs are compared by what they *look like*: matplotlib embeds usetex
    Type1 fonts under a random 6-letter subset prefix (and re-encrypts the
    font program accordingly), so two renders of the identical figure never
    match byte for byte. Hash the pages rasterized at a fixed resolution.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(  # noqa: S603
            ["pdftoppm", "-r", "100", "-png", str(path), f"{tmp}/p"],  # noqa: S607
            check=True,
            capture_output=True,
        )
        h = hashlib.sha256()
        for page in sorted(Path(tmp).glob("p*.png")):
            h.update(page.read_bytes())
        return h.hexdigest()


FLOAT = re.compile(rb"-?\d+\.\d+(?:[eE][-+]?\d+)?")
TEXT_SUFFIXES = {".csv", ".tsv", ".json", ".toml"}


def _normalized_sha256(path: Path) -> str:
    """
    Text outputs hashed with every decimal number rounded to 10 significant
    digits: the same computation summed in a different order (e.g. skbio's
    PERMANOVA statistic) differs in the 15th-16th digit, which is noise, not
    a changed result. Anything that differs at 1e-10 relative still fails.
    """

    def rounded(m: re.Match) -> bytes:
        return format(float(m.group()), ".10g").encode()

    return hashlib.sha256(FLOAT.sub(rounded, path.read_bytes())).hexdigest()


def _features_table_sha256(root: Path) -> str:
    """
    A step that edits config.toml (features.import) is compared by what it
    changes -- the [features] table of the merged configuration -- not by the
    whole file, which also holds unrelated settings that evolve on their own.
    """
    import tomllib

    merged: dict[str, Any] = {}
    for name in ("config.toml", "config.local.toml"):
        path = root / name
        if path.is_file():
            with open(path, "rb") as f:
                features = tomllib.load(f).get("features", {})
            for key, value in features.items():
                merged[key] = value
    payload = json.dumps(merged, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def manifest(case: Case) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rel_out in case.outputs:
        if rel_out == "config.toml":
            out["config.toml[features]"] = {"sha256": _features_table_sha256(SANDBOX), "bytes": 0}
            continue
        root = SANDBOX / rel_out
        if not root.exists():
            continue
        for f in _iter_files(root):
            if f.name in SKIP_NAMES or f.suffix in SKIP_SUFFIXES or f.is_symlink():
                continue
            key = TIMESTAMP.sub("<ts>", str(f.relative_to(SANDBOX)))
            if key in out:
                raise SystemExit(f"[ABORT] two outputs normalize to the same name: {key}")
            if f.suffix == ".pdf":
                out[key] = {
                    "sha256": _pdf_render_sha256(f),
                    "bytes": f.stat().st_size,
                    "rendered": True,
                }
            else:
                out[key] = {"sha256": _sha256(f), "bytes": f.stat().st_size}
                if f.suffix in TEXT_SUFFIXES:
                    out[key]["normalized_sha256"] = _normalized_sha256(f)
    return out


def _keep_small_outputs(case: Case, dest: Path) -> None:
    files = [
        f
        for rel_out in case.outputs
        if (SANDBOX / rel_out).exists()
        for f in _iter_files(SANDBOX / rel_out)
        if f.suffix in KEEP_SUFFIXES and not f.is_symlink()
    ]
    if sum(f.stat().st_size for f in files) > case.keep_limit_mb * 1_000_000:
        return
    if dest.exists():
        shutil.rmtree(dest)
    for f in files:
        target = dest / TIMESTAMP.sub("<ts>", str(f.relative_to(SANDBOX)))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)


def git_commit() -> str:
    try:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(ConfigLoader.PROJECT_ROOT), "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --- running -----------------------------------------------------------------


def run_new(case: Case) -> dict:
    import phantom.stages  # noqa: F401 -- registers the stages
    from phantom.runner.loader import build_jobs, parse_run_doc
    from phantom.runner.planner import execute, plan
    from phantom.runner.stage import RunContext
    from phantom.runner.stamps import StampStore

    build_sandbox(case)
    p = Patcher()
    try:
        with guard_real_inputs(case), sandboxed(), deterministic(p):
            ctx = RunContext.load(SANDBOX / "config.toml")
            run = parse_run_doc({"run": {"name": f"regression-{case.name}"}, "steps": case.steps})
            store = StampStore(ctx.section_path("stamps"))
            execute(plan(build_jobs(run, ctx), ctx, store, force=True), ctx, store, run)
            return manifest(case)
    finally:
        p.close()


def baseline_path(case: Case) -> Path:
    return BASELINES / f"{case.name}.json"


def record_baseline(case: Case, repeat: bool) -> None:
    """Records the current code's output as the new reference -- only after an
    intended change of results, from a trusted commit."""
    first = run_new(case)
    if not first:
        raise SystemExit(f"[ERROR] {case.name}: the run produced no outputs")
    _keep_small_outputs(case, BASELINES / case.name)
    if repeat:
        second = run_new(case)
        diff = compare(first, second)
        if diff:
            print(f"[NONDETERMINISTIC] {case.name}: two runs differ:")
            print(format_diff(diff))
            raise SystemExit(1)
    BASELINES.mkdir(parents=True, exist_ok=True)
    baseline_path(case).write_text(
        json.dumps(
            {
                "case": case.name,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "git_commit": git_commit(),
                "determinism_checked": repeat,
                "files": first,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    total = sum(v["bytes"] for v in first.values())
    print(f"[BASELINE] {case.name}: {len(first)} files, {total / 1e6:.1f} MB")
    _report_real(case, first)
    shutil.rmtree(SANDBOX, ignore_errors=True)


def check(case: Case) -> bool:
    if not baseline_path(case).is_file():
        raise SystemExit(f"[ERROR] {case.name}: no baseline recorded")
    expected = json.loads(baseline_path(case).read_text(encoding="utf-8"))["files"]
    actual = run_new(case)
    diff = compare(expected, actual)
    if diff:
        print(f"[REGRESSION] {case.name}:")
        print(format_diff(diff))
        _keep_small_outputs(case, FAILED / case.name)
        print(f"  new outputs kept in {SANDBOX} (small text files also in {FAILED / case.name})")
        return False
    print(f"[MATCH] {case.name}: {len(actual)} files identical to the baseline")
    shutil.rmtree(SANDBOX, ignore_errors=True)
    return True


def compare(expected: dict, actual: dict) -> dict[str, list[str]]:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))

    def same(e: dict, a: dict) -> bool:
        if e["sha256"] == a["sha256"]:
            return True
        norm = e.get("normalized_sha256")  # absent in baselines recorded before it existed
        return norm is not None and norm == a.get("normalized_sha256")

    changed = sorted(k for k in set(expected) & set(actual) if not same(expected[k], actual[k]))
    return {k: v for k, v in {"missing": missing, "extra": extra, "changed": changed}.items() if v}


def format_diff(diff: dict[str, list[str]]) -> str:
    lines = []
    for kind, names in diff.items():
        lines.append(f"  {kind} ({len(names)}):")
        lines += [f"    {n}" for n in names[:20]]
        if len(names) > 20:
            lines.append(f"    ... {len(names) - 20} more")
    return "\n".join(lines)


def _report_real(case: Case, recorded: dict) -> None:
    """Informational: does the recorded output equal what is on disk today?"""
    for out_rel, real_rel in case.real.items():
        real_root = WORKSPACE / real_rel
        if not real_root.exists():
            continue
        same = differ = 0
        for key, entry in recorded.items():
            if not key.startswith(out_rel):
                continue
            real_file = real_root / key[len(out_rel) :].lstrip("/")
            if real_file.is_file() and _sha256(real_file) == entry["sha256"]:
                same += 1
            else:
                differ += 1
        print(f"  [vs real {real_rel}] identical: {same}, different/absent: {differ}")
