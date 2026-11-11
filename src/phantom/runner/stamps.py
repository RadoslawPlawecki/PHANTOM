"""
Job stamps: the record that lets ``phantom run`` skip work that is already
up to date.

A stamp is keyed by stage + a hash of the job's parameters, and stores a
fingerprint of the job's inputs and the outputs it produced. A job is
*done* when a stamp for its exact parameters exists, its inputs still have
the same fingerprint, and its recorded outputs still exist; otherwise it
has to run.

Fingerprints use (relative path, size, mtime) rather than content hashes:
cheap on multi-GB inputs, and any rewrite of a file changes its mtime.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from phantom.config.loader import ConfigLoader


def params_dict(p: Any) -> dict[str, Any]:
    return dataclasses.asdict(p)


def params_digest(p: Any) -> str:
    payload = json.dumps(params_dict(p), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rel_to_workspace(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ConfigLoader.DATA_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _abs(rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else ConfigLoader.DATA_ROOT / path


def fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=str):
        if not path.exists():
            h.update(f"{rel_to_workspace(path)}|missing\n".encode())
            continue
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            st = file.stat()
            h.update(f"{rel_to_workspace(file)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


@dataclass
class Status:
    state: str  # "done" | "run"
    reason: str
    stamp: dict | None = None


class StampStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path_for(self, stage: str, digest: str) -> Path:
        return self.root / stage / f"{digest}.json"

    def read(self, stage: str, digest: str) -> dict | None:
        path = self.path_for(stage, digest)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def status(self, stage: str, p: Any, inputs: list[Path]) -> Status:
        stamp = self.read(stage, params_digest(p))
        if stamp is None:
            return Status("run", "no previous run with these parameters")
        if stamp.get("inputs_fingerprint") != fingerprint(inputs):
            return Status("run", f"inputs changed since {stamp.get('finished_at', '?')}", stamp)
        missing = [o for o in stamp.get("outputs", []) if not _abs(o).exists()]
        if missing:
            return Status("run", f"previous output missing: {missing[0]}", stamp)
        outputs = stamp.get("outputs", [])
        where = f" -> {outputs[0]}" if len(outputs) == 1 else f" -> {len(outputs)} outputs"
        return Status("done", f"up to date ({stamp.get('finished_at', '?')}){where}", stamp)

    def write(
        self,
        stage: str,
        p: Any,
        inputs: list[Path],
        outputs: list[Path],
        started_at: datetime,
        extra: dict | None = None,
    ) -> Path:
        digest = params_digest(p)
        path = self.path_for(stage, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": stage,
            "params": params_dict(p),
            "params_digest": digest,
            "inputs": [rel_to_workspace(i) for i in inputs],
            "inputs_fingerprint": fingerprint(inputs),
            "outputs": [rel_to_workspace(o) for o in outputs],
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            **(extra or {}),
        }
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return path
