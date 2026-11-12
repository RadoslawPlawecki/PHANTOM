"""
The sample sheet: the user's assemblies under any file names, and the class
label of each. Every sample gets a pipeline id -- S1, S2, ... in the sheet's
row order, or the id its ``id`` column gives -- and its assembly is linked as
``<assemblies>/S<n>.fa``, so every tool output is named after that id
(``S1_geN``, ``S1_geN.fna``, ...). The id, the original file and the label
are recorded in the metadata CSV.

    assembly,label
    /data/run7/patient_A/final.contigs.fa,1
    controls/ctrl_03.fasta,0          (relative = next to the sheet)

Ids must never silently point at another file: the tool outputs of a sample
are kept between runs, so reusing its id for a different assembly would mix
two samples. A sheet that would do that is an error.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ID_RE = re.compile(r"^S(\d+)$")
ASSEMBLY, LABEL, ID = "assembly", "label", "id"
COMPRESSED = (".gz", ".bz2", ".xz", ".zip")


@dataclass(frozen=True)
class Sample:
    id: str
    assembly: Path
    label: int
    explicit_id: bool = False  # from the sheet's id column, not numbered


def _row(i: int) -> str:
    return f"row {i + 2}"  # line 1 is the header


def read_sheet(path: Path) -> tuple[list[Sample], list[str]]:
    """The samples of a sheet, with their ids; and every problem found."""
    if not path.is_file():
        return [], [f"sample sheet not found: {path} (columns: assembly,label)"]
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        return [], [f"{path}: cannot be read as CSV ({e})"]
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in (ASSEMBLY, LABEL) if c not in df.columns]
    if missing:
        return [], [
            f"{path}: missing column(s) {', '.join(missing)} (columns: assembly,label[,id])"
        ]
    if df.empty:
        return [], [f"{path}: no samples"]

    errors: list[str] = []
    files: list[Path] = []
    labels: list[int] = []
    for i, row in enumerate(df.to_dict("records")):
        raw = row[ASSEMBLY].strip()
        file = Path(raw).expanduser()
        file = (file if file.is_absolute() else path.parent / file).resolve()
        if not raw:
            errors.append(f"{path} {_row(i)}: empty assembly")
        elif file.name.endswith(COMPRESSED):
            errors.append(f"{path} {_row(i)}: {raw} is compressed -- decompress it first")
        elif not file.is_file():
            errors.append(f"{path} {_row(i)}: assembly not found: {file}")
        files.append(file)
        label = row[LABEL].strip()
        if label not in ("0", "1"):
            errors.append(f"{path} {_row(i)}: label must be 0 (control) or 1 (case), got {label!r}")
        labels.append(1 if label == "1" else 0)

    seen: dict[Path, int] = {}
    for i, file in enumerate(files):
        if file in seen:
            errors.append(f"{path} {_row(i)}: {file} is already listed in {_row(seen[file])}")
        seen.setdefault(file, i)

    given = df[ID].str.strip().tolist() if ID in df.columns else [""] * len(df)
    ids, id_errors = _assign_ids(given, path)
    errors += id_errors
    if errors:
        return [], errors
    samples = [
        Sample(sid, f, lab, explicit_id=bool(g))
        for sid, g, f, lab in zip(ids, given, files, labels, strict=True)
    ]
    return samples, []


def _assign_ids(given: list[str], path: Path) -> tuple[list[str], list[str]]:
    """Ids from the ``id`` column; blank ones (or no column) are numbered on
    after the highest given id, in row order."""
    errors = []
    numbers = []
    for i, sid in enumerate(given):
        m = ID_RE.match(sid)
        if m:
            numbers.append(int(m.group(1)))
        elif sid:
            errors.append(f"{path} {_row(i)}: id must look like S1, S2, ..., got {sid!r}")
    dupes = sorted({s for s in given if s and given.count(s) > 1})
    if dupes:
        errors.append(f"{path}: duplicate id(s) {', '.join(dupes)}")
    nxt = max(numbers, default=0) + 1
    ids = []
    for sid in given:
        if not sid:
            sid, nxt = f"S{nxt}", nxt + 1
        ids.append(sid)
    return ids, errors


def check_against_metadata(
    samples: list[Sample], metadata: pd.DataFrame, accept_changes: bool = False
) -> list[str]:
    """
    Problems with applying the sheet to a workspace that already has samples:
    an id whose assembly changed, or a sample the metadata has (it has tool
    outputs) that the sheet no longer lists.
    """
    if metadata.empty or ID not in metadata.columns:
        return []
    by_id = {s.id: s for s in samples}
    recorded = (
        dict(zip(metadata[ID], metadata[ASSEMBLY], strict=True))
        if ASSEMBLY in metadata.columns
        else {}
    )
    errors = []
    dropped = [sid for sid in metadata[ID] if sid not in by_id]
    if dropped:
        errors.append(
            f"sample(s) {', '.join(dropped)} are in the metadata (they have tool outputs) but not "
            "in the sample sheet -- removing samples is not supported; keep them in the sheet"
        )
    unrecorded = [
        s.id
        for s in samples
        if not s.explicit_id
        and s.id in set(metadata[ID])
        and not isinstance(recorded.get(s.id), str)
    ]
    if unrecorded:
        errors.append(
            f"sample(s) {', '.join(unrecorded)} already exist in the metadata without a recorded "
            "assembly (from before sample sheets): give the sheet an id column so every "
            "assembly is matched to its existing id explicitly"
        )
    for sid, before in recorded.items():
        if accept_changes or sid not in by_id or not isinstance(before, str):
            continue
        if before != str(by_id[sid].assembly):
            errors.append(
                f"{sid} was {before}, the sheet now makes it {by_id[sid].assembly}. Ids "
                "follow the sheet's row order: add new samples at the end, or give an id "
                "column; set accept_changes = true if the file only moved"
            )
    return errors


def link_assemblies(samples: list[Sample], directory: Path) -> list[Path]:
    """``<directory>/S<n>.fa`` -> each sample's assembly; stale links removed."""
    directory.mkdir(parents=True, exist_ok=True)
    wanted = {f"{s.id}.fa": s for s in samples}
    for old in directory.glob("S*.fa"):
        if old.is_symlink() and ID_RE.match(old.stem) and old.name not in wanted:
            old.unlink()
    links = []
    for name, sample in wanted.items():
        link = directory / name
        if link.is_symlink():
            if Path(os.readlink(link)) == sample.assembly:
                links.append(link)
                continue
            link.unlink()
        elif link.exists():
            if link.resolve() == sample.assembly:
                links.append(link)
                continue
            raise FileExistsError(
                f"{link} is a regular file, not the link this step manages; move it away"
            )
        link.symlink_to(sample.assembly)
        links.append(link)
    return links


def metadata_frame(samples: list[Sample]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID: [s.id for s in samples],
            ASSEMBLY: [str(s.assembly) for s in samples],
            LABEL: [s.label for s in samples],
        }
    )
