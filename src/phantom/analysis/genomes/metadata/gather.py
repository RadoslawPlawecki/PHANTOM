"""
Gathering sample metadata into the merged metadata CSV, without prompting:
class labels (through a LabelingStrategy) and the statistics scraped from
tool outputs (runtimes, CheckV quality counts, assembly sizes).

Column groups are appended in a fixed order -- labels, runtimes, CheckV,
sizes -- so the CSV's layout does not depend on how a run was requested.
Scraped statistics are facts derived from the tool outputs, so gathering
them again replaces the previous values; labels are a decision, so existing
labels are only replaced when explicitly asked to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from phantom.analysis.genomes.metadata.labeling import (
    ExternalCSVLabeler,
    LabelingStrategy,
    ManualLabeler,
    PatternLabeler,
    SampleThresholdLabeler,
)
from phantom.analysis.genomes.metadata.mapping import MetadataManager

LABEL_COLUMN = "label"
SCAN_COLUMNS = {
    "runtimes": ["vib_runtime", "vs2_runtime", "gen_runtime"],
    "checkv": [
        "checkv_vib_total",
        "checkv_vib_medium",
        "checkv_vib_high",
        "checkv_vs2_total",
        "checkv_vs2_medium",
        "checkv_vs2_high",
        "checkv_gen_total",
        "checkv_gen_medium",
        "checkv_gen_high",
    ],
    "sizes": ["megahit_size_bytes", "megahit_size_mb"],
}
TARGETS = ("labeling", *SCAN_COLUMNS)
STRATEGIES = {
    "threshold": ("threshold",),
    "manual": ("positives",),
    "pattern": ("pattern",),
    "csv": ("path",),
}


def check_labeling_spec(spec: dict[str, Any]) -> list[str]:
    """Problems with a run file's ``labeling = { strategy = ..., ... }`` table."""
    strategy = spec.get("strategy")
    if strategy not in STRATEGIES:
        return [f"labeling.strategy must be one of {', '.join(STRATEGIES)}, got {strategy!r}"]
    errors = [
        f"labeling: strategy {strategy!r} needs {key!r}"
        for key in STRATEGIES[strategy]
        if key not in spec
    ]
    allowed = {"strategy", *STRATEGIES[strategy]} | (
        {"join_key", "label_col"} if strategy == "csv" else set()
    )
    errors += [
        f"labeling: unknown key {k!r} for strategy {strategy!r}" for k in spec if k not in allowed
    ]
    if strategy == "threshold" and not isinstance(spec.get("threshold"), int):
        errors.append("labeling.threshold must be an integer sample number")
    if strategy == "manual" and not isinstance(spec.get("positives"), list):
        errors.append('labeling.positives must be a list of sample ids, e.g. ["S1", "S7"]')
    return errors


def build_labeler(spec: dict[str, Any], resolve) -> LabelingStrategy:
    strategy = spec["strategy"]
    if strategy == "threshold":
        return SampleThresholdLabeler(spec["threshold"])
    if strategy == "manual":
        return ManualLabeler(spec["positives"])
    if strategy == "pattern":
        return PatternLabeler(spec["pattern"])
    kwargs = {k: spec[k] for k in ("join_key", "label_col") if k in spec}
    return ExternalCSVLabeler(Path(resolve(spec["path"])), **kwargs)


def gather_metadata(
    config: dict,
    meta: MetadataManager,
    targets: list[str],
    labeler: LabelingStrategy | None = None,
    overwrite_labels: bool = False,
) -> list[str]:
    """Adds the requested column groups to the metadata CSV; returns the
    columns written."""
    from phantom.analysis.genomes.metadata.scanner import scan_metadata

    current = meta.load()
    if current.empty:
        raise ValueError(f"no samples in {meta.path}; run preprocessing.collect first")
    parts: list[tuple[pd.DataFrame, set[str]]] = []

    if "labeling" in targets:
        if labeler is None:
            raise ValueError("targets include 'labeling' but no labeling strategy was given")
        if LABEL_COLUMN in current.columns and not overwrite_labels:
            raise ValueError(
                f"{meta.path} already has labels; set overwrite_labels = true to replace them"
            )
        labels = labeler.assign_labels(current)
        df_labels = pd.DataFrame(list(labels.items()), columns=["id", LABEL_COLUMN])
        positives = int(df_labels[LABEL_COLUMN].sum())
        print(f"[INFO] Labeled {len(df_labels)} samples ({positives} positives).")
        parts.append((df_labels, {LABEL_COLUMN}))

    scan_targets = [t for t in SCAN_COLUMNS if t in targets]
    if scan_targets:
        print("[INFO] Scanning logs, sizes and CheckV results...")
        df_scan = scan_metadata(config)
        if not df_scan.empty:
            for target in scan_targets:
                cols = [c for c in SCAN_COLUMNS[target] if c in df_scan.columns]
                if cols:
                    parts.append((df_scan[["id", *cols]], set(cols)))
                else:
                    print(f"[WARNING] Nothing found for {target!r}.")

    written = []
    for df_part, columns in parts:
        meta.add_metadata_columns(df_part, overwrite=columns)
        written += sorted(columns)
    return written


def collect_genomes(config: dict, meta: MetadataManager) -> pd.DataFrame:
    """
    Copies the identification tools' viral contigs into the genome registry
    and records the original/formatted file names (the "renaming" columns)
    in the metadata CSV, replacing any from an earlier collection.
    """
    from phantom.analysis.genomes.collection import GenomeCollector

    collector = GenomeCollector(config, meta)
    invalid = collector.scan_for_invalid()
    if invalid:
        print(
            f"[WARNING] {invalid} folder(s) not named S<n>_<tool> skipped: tool outputs must "
            "come from assemblies numbered by preprocessing.samples."
        )
    df = collector.collect()
    if df.empty:
        raise ValueError("no genomes were found or copied")
    meta.add_metadata_columns(df, overwrite=set(df.columns) - {"id"})
    return df
