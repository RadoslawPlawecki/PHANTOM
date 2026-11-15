"""
The raw assemblies -> pipeline samples: ids, renamed assemblies, labels.

The user puts one assembly per sample into the raw folder (``[raw].path``),
in one subfolder per class -- the folder gives the label:

    data/raw/
    ├── case/                          label 1 (or the [classes] case name, e.g. allergy/)
    │   ├── patient_A.fa
    │   └── patient_B/final.contigs.fa a folder per sample: its one FASTA file
    └── control/                       label 0 (or e.g. healthy/)
        └── ctrl-03.fasta

Every sample gets an id S1, S2, ...: new samples are numbered in path order
after the highest id so far, and a sample named ``S<n>`` (file or folder)
keeps that id. Its assembly is linked as ``<assemblies>/S<n>.fa`` -- the
renamed assembly every later step reads, so every tool output is named after
the id. The mapping id -> assembly -> label goes to the sample mapping
(``[samples].path``) and the metadata CSV.

Ids never move to another file: the tool outputs of a sample are kept by id.
A sample is recognized by its path inside its class folder, so moving it to
the other class folder relabels it; a sample that has an id but is no longer
found is an error (removing samples is not supported).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

from phantom.config.classes import DEFAULTS, slug

ID_RE = re.compile(r"^S(\d+)$")
ID, ASSEMBLY, LABEL = "id", "assembly", "label"
FASTA = (".fa", ".fasta", ".fna", ".fas")
COMPRESSED = (".gz", ".bz2", ".xz", ".zip")
LABELS = {"control": 0, "case": 1}


@dataclass(frozen=True)
class Sample:
    id: str
    assembly: Path  # the raw file
    relative: str  # its path in the raw folder, e.g. "case/patient_A.fa"
    label: int

    @property
    def key(self) -> str:
        """What identifies the sample: its path inside the class folder."""
        return _key(self.relative)


def _key(relative: str) -> str:
    return str(PurePosixPath(*PurePosixPath(relative).parts[1:]))


def _id_number(sample_id: str) -> int:
    match = ID_RE.match(sample_id)
    return int(match.group(1)) if match else 0


def _natural(text: str) -> list:
    """Sort key: "patient2" before "patient10"."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _is_fasta(path: Path) -> bool:
    return path.suffix.lower() in FASTA


def _is_compressed_fasta(path: Path) -> bool:
    return path.suffix.lower() in COMPRESSED and Path(path.stem).suffix.lower() in FASTA


def _visible(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.iterdir() if not p.name.startswith(".")), key=lambda p: p.name
    )


def class_folders(names: dict[str, str]) -> tuple[dict[str, int], list[str]]:
    """
    Folder name (as a slug) -> label: ``case``/``control`` and the class names
    of ``[classes]`` (``names``: {"control": ..., "case": ...}), any case.
    """
    folders: dict[str, int] = {}
    errors = []
    for key, label in LABELS.items():
        for name in {key, names.get(key, DEFAULTS[key])}:
            folder = slug(name)
            if folders.get(folder, label) != label:
                errors.append(
                    f"[classes]: {name!r} names both classes as a folder; give them distinct names"
                )
            folders[folder] = label
    return folders, errors


def scan_raw(raw: Path, folders: dict[str, int]) -> tuple[list[tuple[str, Path, int]], list[str]]:
    """
    The samples in the raw folder: (path in the raw folder, file, label), in
    path order; and every problem found.
    """
    expected = "one subfolder per class: case/ and control/"
    if not raw.is_dir():
        return [], [f"raw folder not found: {raw} ({expected})"]
    found: list[tuple[str, Path, int]] = []
    errors = []
    for entry in _visible(raw):
        if entry.is_file():
            if _is_fasta(entry) or _is_compressed_fasta(entry):
                errors.append(
                    f"{entry}: put it into a class folder ({raw / 'case'} or {raw / 'control'})"
                )
            continue
        label = folders.get(slug(entry.name))
        if label is None:
            errors.append(
                f"{entry}: unknown class folder {entry.name!r} -- use case/ or control/ "
                "(or the names of [classes])"
            )
            continue
        for sample in _visible(entry):
            files, problems = _assembly_of(sample)
            errors += problems
            found += [(f.relative_to(raw).as_posix(), f.resolve(), label) for f in files]
    if not found and not errors:
        errors.append(f"no assemblies in {raw} ({expected}, one FASTA file per sample)")
    keys: dict[str, str] = {}
    for relative, _, _ in found:
        other = keys.setdefault(_key(relative), relative)
        if other != relative:
            errors.append(
                f"{raw}: {other} and {relative} are the same sample name in both classes; "
                "rename one"
            )
    found.sort(key=lambda sample: _natural(sample[0]))
    return found, errors


def _assembly_of(entry: Path) -> tuple[list[Path], list[str]]:
    """The assembly of one entry of a class folder: a FASTA file, or a folder
    holding exactly one (non-FASTA files are ignored)."""
    if entry.is_file():
        if _is_compressed_fasta(entry):
            return [], [f"{entry} is compressed -- decompress it first (e.g. gunzip)"]
        return ([entry], []) if _is_fasta(entry) else ([], [])
    files = sorted(
        p
        for p in entry.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(entry).parts)
    )
    compressed = [p for p in files if _is_compressed_fasta(p)]
    fasta = [p for p in files if _is_fasta(p)]
    if compressed:
        return [], [f"{compressed[0]} is compressed -- decompress it first (e.g. gunzip)"]
    if len(fasta) > 1:
        listed = ", ".join(p.relative_to(entry).as_posix() for p in fasta)
        return [], [
            f"{entry}: one sample folder holds {len(fasta)} FASTA files ({listed}); keep "
            "only its assembly"
        ]
    if not fasta:
        return [], [f"{entry}: no FASTA file ({', '.join(FASTA)}) in this sample folder"]
    return fasta, []


def assign_ids(
    found: list[tuple[str, Path, int]], metadata: pd.DataFrame
) -> tuple[list[Sample], list[str]]:
    """
    Ids for the samples found: the id the metadata recorded for the same
    sample, else the ``S<n>`` the sample is named, else the next free number.
    Samples sorted by id; and every problem (an id that would move to another
    file, a recorded sample that is gone).
    """
    recorded: dict[str, str] = {}  # key -> id
    unrecorded: set[str] = set()  # ids from before the raw folder: no assembly recorded
    if not metadata.empty and ID in metadata.columns:
        assemblies = metadata[ASSEMBLY] if ASSEMBLY in metadata.columns else None
        for i, sample_id in enumerate(metadata[ID]):
            before = assemblies.iloc[i] if assemblies is not None else None
            if isinstance(before, str) and before:
                recorded[_key(before)] = sample_id
            else:
                unrecorded.add(sample_id)
    owner = {sample_id: key for key, sample_id in recorded.items()}

    errors = []
    ids: dict[str, str] = {}  # key -> id
    for relative, _, _ in found:
        key = _key(relative)
        named = PurePosixPath(key).parts[0].split(".")[0]
        if key in recorded:
            ids[key] = recorded[key]
        elif ID_RE.match(named):
            if named in owner:
                errors.append(f"{relative} is named {named}, but {named} is {owner[named]}")
            elif named in ids.values():
                errors.append(f"{relative}: two samples are named {named}")
            ids[key] = named
    keys = {_key(relative) for relative, _, _ in found}
    gone = [f"{sample_id} ({key})" for key, sample_id in recorded.items() if key not in keys]
    if gone:
        errors.append(
            f"sample(s) {', '.join(gone)} not found in the raw folder -- removing samples is "
            "not supported (their tool outputs exist): put them back"
        )
    orphans = sorted(unrecorded - set(ids.values()), key=_id_number)
    if orphans:
        errors.append(
            f"sample(s) {', '.join(orphans)} are in the metadata without a recorded assembly "
            "(from before the raw folder): name their files or folders S<n> so each keeps its id"
        )
    following = max((_id_number(i) for i in [*ids.values(), *owner, *unrecorded]), default=0) + 1
    for relative, _, _ in found:
        if _key(relative) not in ids:
            ids[_key(relative)] = f"S{following}"
            following += 1
    if errors:
        return [], errors
    samples = [
        Sample(ids[_key(relative)], file, relative, label) for relative, file, label in found
    ]
    return sorted(samples, key=lambda s: _id_number(s.id)), []


def read_samples(
    raw: Path, names: dict[str, str], metadata: pd.DataFrame
) -> tuple[list[Sample], list[str]]:
    """The samples of the raw folder with their ids, or every problem found."""
    folders, errors = class_folders(names)
    if errors:
        return [], errors
    found, errors = scan_raw(raw, folders)
    if errors:
        return [], errors
    return assign_ids(found, metadata)


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
    """id, assembly (its path in the raw folder), label -- the sample mapping."""
    return pd.DataFrame(
        {
            ID: [s.id for s in samples],
            ASSEMBLY: [s.relative for s in samples],
            LABEL: [s.label for s in samples],
        }
    )


def write_mapping(samples: list[Sample], path: Path) -> bool:
    """The sample mapping CSV; rewritten only when it changes (True then)."""
    text = metadata_frame(samples).to_csv(index=False)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True
