"""
Which outputs of the example pipeline are kept as reference ("golden") results,
how they are made comparable (no temporary paths, no run timestamps), and how
two copies are compared (numbers with a tolerance, so a different CPU or
library build does not count as a change).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

# One or more outputs of every pipeline step, relative to the example workspace.
SELECTED = [
    "data/samples.csv",
    "data/genomes/genome_metadata.csv",
    "data/masks/*/*.tsv",
    "data/features/1.0/extracted/*/*.csv",
    "results/diversity/1.0/*.csv",
    "results/figures/1.0/*/metrics.csv",
    "results/mds/1.0/*/unsupervised_metrics.csv",
    "results/optimization/1.0/**/*_result.json",
    "results/optimization/1.0/**/permutation_test.json",
    "results/optimization/1.0/**/nested/*/metrics.json",
]
TIMESTAMP = re.compile(r"\d{8}_\d{6}")
RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-9


# Derived reference tables: one small table standing for many output files.
FUNCTION_SUMMARY = "data/functions/1.0/orf_counts.csv"


def _run_name(path: Path, root: Path) -> str:
    """The output's name with its run folder's timestamp replaced by what tells
    the runs apart (MDS run folders carry no tool name: it is in metadata.json)."""
    name = str(path.relative_to(root))
    for parent in path.parents:
        if TIMESTAMP.search(parent.name) and (parent / "metadata.json").is_file():
            metadata = json.loads((parent / "metadata.json").read_text())
            tool = metadata.get("vtool") or Path(str(metadata.get("comp", ""))).name.split("_")[0]
            return name.replace(parent.name, TIMESTAMP.sub(tool or "<ts>", parent.name))
    return TIMESTAMP.sub("<ts>", name)


def selected_outputs(root: Path) -> dict[str, Path]:
    """name (relative path, run timestamps replaced) -> file, for every selected output."""
    files = {}
    for pattern in SELECTED:
        for path in sorted(root.glob(pattern)):
            name = _run_name(path, root)
            if name in files:
                raise ValueError(f"two outputs would share the reference name {name}")
            files[name] = path
    return files


def function_summary(root: Path) -> pd.DataFrame:
    """features.summarize_functions writes one file per (tool, function, sample):
    the number of ORFs in each stands for them all."""
    rows = [
        {
            "file": str(path.relative_to(root / "data" / "functions" / "1.0")),
            "orfs": sum(1 for line in path.read_text().splitlines()[1:] if line.strip()),
        }
        for path in sorted((root / "data" / "functions" / "1.0").glob("*/*/*.tsv"))
    ]
    return pd.DataFrame(rows, columns=["file", "orfs"])


def comparable_text(path: Path, root: Path) -> str:
    """The file's text with the workspace path and run timestamps masked."""
    text = path.read_text(encoding="utf-8").replace(str(root.resolve()), "<workspace>")
    return TIMESTAMP.sub("<ts>", text.replace(str(root), "<workspace>"))


def _read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix == ".tsv" else None
    return pd.read_csv(path, sep=separator, engine="python")


def differences(expected: Path, actual: Path) -> list[str]:
    """What differs between a golden file and a new output (empty: the same)."""
    if expected.suffix == ".json":
        return _json_differences(json.loads(expected.read_text()), json.loads(actual.read_text()))
    old, new = _read_table(expected), _read_table(actual)
    if list(old.columns) != list(new.columns):
        return [f"columns {list(old.columns)} -> {list(new.columns)}"]
    if len(old) != len(new):
        return [f"{len(old)} rows -> {len(new)} rows"]
    problems = []
    for column in old.columns:
        a, b = old[column], new[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            for row, (x, y) in enumerate(zip(a, b, strict=True)):
                if not _close(x, y):
                    problems.append(f"{column}, row {row + 1}: {x} -> {y}")
        elif not a.astype(str).equals(b.astype(str)):
            problems.append(f"{column}: values differ")
    return problems


def _close(x, y) -> bool:
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return math.isclose(x, y, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)


def _json_differences(old, new, where: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        if set(old) != set(new):
            return [f"{where or 'top level'}: keys {sorted(old)} -> {sorted(new)}"]
        return [p for key in old for p in _json_differences(old[key], new[key], f"{where}.{key}")]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{where}: {len(old)} items -> {len(new)}"]
        return [
            p
            for i, (a, b) in enumerate(zip(old, new, strict=True))
            for p in _json_differences(a, b, f"{where}[{i}]")
        ]
    numbers = (int, float)
    if isinstance(old, numbers) and isinstance(new, numbers) and not isinstance(old, bool):
        return [] if _close(old, new) else [f"{where}: {old} -> {new}"]
    return [] if old == new else [f"{where}: {old!r} -> {new!r}"]
