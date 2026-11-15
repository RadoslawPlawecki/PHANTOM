"""
Scans output files from geNomad, VIBRANT, and VirSorter2 to extract runtime
information.
"""

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from phantom.config.loader import ConfigLoader

TIME_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")
VIBRANT_RE = re.compile(r"Runtime:\s*([0-9.]+)\s*minutes")


def get_sample_id(path):
    name = Path(path).name
    match = re.search(r"S\d+", name)
    if match:
        return match.group(0)
    raise ValueError(f"Cannot extract sample_id from {name}")


def get_file_size(path):
    size_bytes = path.stat().st_size
    return size_bytes, round(size_bytes / (1024 * 1024), 2)


def extract_first_time(file_path):
    with open(file_path) as f:
        for line in f:
            match = TIME_RE.search(line)
            if match:
                return match.group(1)
    return None


def extract_last_time(file_path):
    last = None
    with open(file_path) as f:
        for line in f:
            match = TIME_RE.search(line)
            if match:
                last = match.group(1)
    return last


def get_virsorter2_log_time(log_file):
    with open(log_file) as f:
        first_line = f.readline()
    start_str = first_line.split("]")[0].strip("[")
    start_str = " ".join(start_str.split()[:2])
    return datetime.strptime(start_str, "%Y-%m-%d %H:%M")


def count_checkv_scored(path) -> int | None:
    """Contigs of a CheckV quality summary with a completeness and a
    contamination estimate (None: not a quality summary)."""
    total = 0
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = {k.lower(): k for k in reader.fieldnames or []}
        comp_key = fields.get("completeness") or fields.get("estimated_completeness")
        cont_key = fields.get("contamination") or fields.get("estimated_contamination")
        if not comp_key or not cont_key:
            return None
        for row in reader:
            try:
                float(row[comp_key])
                float(row[cont_key])
            except (ValueError, TypeError):
                continue
            total += 1
    return total


MASK_COUNTS = {"medium": "MQ_provMAGs.tsv", "high": "HQ_provMAGs.tsv"}


def count_masked(masks_dir: Path, tool: str) -> dict[str, dict[str, int]]:
    """
    Medium/high-quality contigs per sample, counted from the masks
    ``preprocessing.masks`` wrote -- so the counts always use the thresholds
    the masks were made with. A missing mask file is an error.
    """
    counts = {}
    for quality, filename in MASK_COUNTS.items():
        path = masks_dir / tool / filename
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found -- run preprocessing.masks first")
        ids = pd.read_csv(path, sep="\t", usecols=["contig_id"])["contig_id"]
        samples = ids.str.split("|", n=1).str[1].str.split("_").str[0]
        counts[quality] = samples.value_counts().to_dict()
    return counts


def build_index(root):
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        print(
            f"[WARNING] Tool directory not found: '{root_path}'. Skipping runtimes for this path."
        )
        return {}
    return {get_sample_id(f): f for f in root_path.iterdir() if f.is_dir()}


def build_assembly_index(root):
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        print(f"[WARNING] Assembly directory not found: '{root_path}'. Skipping size metrics.")
        return {}
    # S<n>.fa, linked by preprocessing.samples (stat() follows the link: real sizes)
    return {f.stem: f for f in root_path.glob("S*.fa") if re.fullmatch(r"S\d+", f.stem)}


def build_checkv_index(root):
    root_path = Path(root)
    index = {}
    if not root_path.exists() or not root_path.is_dir():
        print(
            f"[WARNING] CheckV directory not found: '{root_path}'. "
            "Skipping quality control metrics for this tool."
        )
        return index
    for folder in root_path.iterdir():
        if folder.is_dir():
            tsv = folder / "quality_summary.tsv"
            if tsv.exists():
                index[get_sample_id(folder)] = tsv
    return index


def runtime_vibrant(file_path: Path):
    with open(file_path) as f:
        for line in f:
            match = VIBRANT_RE.search(line)
            if match:
                return round(float(match.group(1)))
    return None


def runtime_virsorter2(folder: Path):
    start_file = folder / "log" / "iter-0" / "step1-pp" / "circular-remove-partial-gene-common.log"
    end_file = (
        folder
        / "log"
        / "iter-0"
        / "step2-extract-feature"
        / "extract-feature-from-hmmout-common.log"
    )
    if not start_file.exists() or not end_file.exists():
        return None
    start_time = get_virsorter2_log_time(start_file)
    end_time = get_virsorter2_log_time(end_file)
    if end_time < start_time:
        end_time += timedelta(days=1)
    return round((end_time - start_time).total_seconds() / 60)


def runtime_genomad(folder):
    start_files = list(folder.glob("*_annotate.log"))
    end_files = list(folder.glob("*_summary.log"))
    if not start_files or not end_files:
        return None
    start = datetime.strptime(extract_first_time(start_files[0]), "%H:%M:%S")
    end = datetime.strptime(extract_last_time(end_files[0]), "%H:%M:%S")
    if end < start:
        end += timedelta(days=1)
    return round((end - start).total_seconds() / 60)


def scan_metadata(config: dict, checkv: bool = True) -> pd.DataFrame:
    """
    Scans predefined roots for metadata and formats it as a DataFrame
    joined on id.

    NOTE: Log parsing for runtimes are strictly hardcoded for the utilized
    versions of geNomad, VIBRANT and VirSorter2. Custom tools added to the
    config are ignored during this step due to the specific nature of various
    outputs.
    """
    path = ConfigLoader.config_path
    vib_path = path(config, "vibrant")
    vs2_path = path(config, "virsorter2")
    gen_path = path(config, "genomad")
    assembly_path = path(config, "assemblies")
    checkv_base = path(config, "checkv")

    vib_root = build_index(vib_path)
    vs2_root = build_index(vs2_path)
    gen_root = build_index(gen_path)
    assembly_root = build_assembly_index(assembly_path)

    checkv_vib = build_checkv_index(checkv_base / "vibrant")
    checkv_vs2 = build_checkv_index(checkv_base / "virsorter2")
    checkv_gen = build_checkv_index(checkv_base / "genomad")

    sample_ids = set(vib_root) | set(vs2_root) | set(gen_root) | set(assembly_root)
    results = {sample_id: {"id": sample_id} for sample_id in sample_ids}

    for sample_id, folder in vib_root.items():
        logs = list(folder.rglob("VIBRANT_log_run_*.log"))
        if logs:
            results[sample_id]["vib_runtime"] = runtime_vibrant(
                max(logs, key=lambda f: f.stat().st_mtime)
            )
    for sample_id, folder in vs2_root.items():
        results[sample_id]["vs2_runtime"] = runtime_virsorter2(folder)
    for sample_id, folder in gen_root.items():
        results[sample_id]["gen_runtime"] = runtime_genomad(folder)

    for sample_id, file in assembly_root.items():
        b, mb = get_file_size(file)
        results[sample_id]["raw_size_bytes"] = b
        results[sample_id]["raw_size_mb"] = mb

    if not checkv:
        return pd.DataFrame(list(results.values()))
    masks_dir = path(config, "masks")
    for tag, tool, index in (
        ("vib", "vibrant", checkv_vib),
        ("vs2", "virsorter2", checkv_vs2),
        ("gen", "genomad", checkv_gen),
    ):
        if not index:
            continue
        masked = count_masked(masks_dir, tool)
        for sample_id, summary in index.items():
            total = count_checkv_scored(summary)
            if total is not None:
                results[sample_id][f"checkv_{tag}_total"] = total
                results[sample_id][f"checkv_{tag}_medium"] = masked["medium"].get(sample_id, 0)
                results[sample_id][f"checkv_{tag}_high"] = masked["high"].get(sample_id, 0)

    return pd.DataFrame(list(results.values()))
