"""
Checks a loaded configuration (config.toml + config.local.toml) before
anything runs: unknown sections or keys (typos), wrong value types and
inconsistent references, each reported with the field it is about. Every
problem is collected, so one ``phantom validate`` shows them all.

Which sections a run actually needs is checked by the stages themselves
(``validate``); this module checks that whatever *is* configured is well
formed.
"""

from __future__ import annotations

import difflib
from typing import Any

from phantom.config.loader import VIRUS_IDENTIFICATION_TOOLS_SECTION

# Sections that name a location: path (+ type/description) and these extra keys.
LOCATIONS: dict[str, set[str]] = {
    "raw": set(),
    "samples": set(),
    "assemblies": set(),
    "genomad": {"viral_fna"},
    "vibrant": {"viral_fna"},
    "virsorter2": {"viral_fna"},
    "genomes": {"metadata_path"},
    "checkv": set(),
    "masks": set(),
    "phanotate": {"orf_fna", "genes", "bed", "proteins"},
    "phabox2": set(),
    "functions": set(),
    "results": set(),
    "stamps": set(),
}
LOCATION_KEYS = {"path", "type", "description"}
STRING_TABLES = ("databases", "envs", "software")
WORKFLOW_KEYS = {"snakefile", "snakemake", "workdir", "cores"}
MODALITY_KEYS = {"tool", "tag", "label"}
MODALITY_TOOLS = {"cherry", "phagcn", "phavip", "phatyp"}
VERSION_KEYS = {"path", "version", "type", "description"}
TOP_LEVEL = {
    "workspace",
    VIRUS_IDENTIFICATION_TOOLS_SECTION,
    "modalities",
    "workflow",
    "threads",
    "features",
    "classes",
    *LOCATIONS,
    *STRING_TABLES,
}


class ConfigError(ValueError):
    """Every problem found in a configuration."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _unknown(where: str, keys, allowed: set[str]) -> list[str]:
    errors = []
    for key in keys:
        if key in allowed:
            continue
        close = difflib.get_close_matches(key, sorted(allowed), n=1)
        hint = f" -- did you mean {close[0]!r}?" if close else ""
        errors.append(f"{where}: unknown key {key!r}{hint}")
    return errors


def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _table(config: dict, name: str, errors: list[str]) -> dict:
    value = config.get(name, {})
    if not isinstance(value, dict):
        errors.append(f"[{name}] must be a table, got {type(value).__name__}")
        return {}
    return value


def _check_locations(config: dict, errors: list[str]) -> None:
    for name, extra in LOCATIONS.items():
        if name not in config:
            continue
        table = _table(config, name, errors)
        errors += _unknown(f"[{name}]", table, LOCATION_KEYS | extra)
        for key in ("path", *sorted(extra)):
            if key in table and not (_is_str(table[key]) and table[key]):
                errors.append(f"[{name}].{key} must be a non-empty string")


def _check_tools(config: dict, errors: list[str]) -> None:
    where = f"[{VIRUS_IDENTIFICATION_TOOLS_SECTION}]"
    tools = _table(config, VIRUS_IDENTIFICATION_TOOLS_SECTION, errors)
    for abbreviation, name in tools.items():
        if not _is_str(name):
            errors.append(f"{where}.{abbreviation} must be a tool name (string)")
        elif name not in LOCATIONS or "viral_fna" not in LOCATIONS[name]:
            known = sorted(k for k, v in LOCATIONS.items() if "viral_fna" in v)
            errors.append(
                f"{where}.{abbreviation} = {name!r}: not a virus identification tool ({known})"
            )


def _check_modalities(config: dict, errors: list[str]) -> None:
    for key, spec in _table(config, "modalities", errors).items():
        where = f"[modalities].{key}"
        if not isinstance(spec, dict):
            errors.append(f"{where} must be a table {{ tool = ..., tag = ..., label = ... }}")
            continue
        errors += _unknown(where, spec, MODALITY_KEYS)
        missing = sorted(MODALITY_KEYS - set(spec))
        if missing:
            errors.append(f"{where}: missing {', '.join(missing)}")
        if "tool" in spec and spec["tool"] not in MODALITY_TOOLS:
            errors.append(
                f"{where}.tool must be one of {sorted(MODALITY_TOOLS)}, got {spec['tool']!r}"
            )
        for field in ("tag", "label"):
            if field in spec and not (_is_str(spec[field]) and spec[field]):
                errors.append(f"{where}.{field} must be a non-empty string")


def _check_workflow(config: dict, errors: list[str]) -> None:
    workflow = _table(config, "workflow", errors)
    errors += _unknown("[workflow]", workflow, WORKFLOW_KEYS)
    command = workflow.get("snakemake")
    if command is not None and not (
        isinstance(command, list) and command and all(_is_str(c) and c for c in command)
    ):
        errors.append('[workflow].snakemake must be a command as a list, e.g. ["snakemake"]')
    cores = workflow.get("cores")
    if cores is not None and cores != "all" and not _positive_int(cores):
        errors.append(f'[workflow].cores must be "all" or a positive number, got {cores!r}')
    for key in ("snakefile", "workdir"):
        if key in workflow and not (_is_str(workflow[key]) and workflow[key]):
            errors.append(f"[workflow].{key} must be a non-empty string")


def _check_threads(config: dict, errors: list[str]) -> None:
    for key, value in _table(config, "threads", errors).items():
        if not _positive_int(value):
            errors.append(f"[threads].{key} must be a positive number, got {value!r}")


def _check_string_tables(config: dict, errors: list[str]) -> None:
    for name in STRING_TABLES:
        for key, value in _table(config, name, errors).items():
            if not _is_str(value):
                errors.append(f"[{name}].{key} must be a string, got {value!r}")


def _check_features(config: dict, errors: list[str]) -> None:
    from phantom.config.feature_versions import NEW_VERSION_STAGES

    features = _table(config, "features", errors)
    for key, value in features.items():
        if key in LOCATION_KEYS:
            continue
        where = f"[features.{key}]"
        if not isinstance(value, dict):
            errors.append(f"[features].{key}: unknown key (feature versions are tables)")
            continue
        stages = set(NEW_VERSION_STAGES)
        errors += _unknown(where, value, VERSION_KEYS | stages)
        for field in ("path", "version"):
            if not (_is_str(value.get(field)) and value.get(field)):
                errors.append(f"{where}.{field} must be a non-empty string")
        for stage in sorted(stages & set(value)):
            table = value[stage]
            if not (isinstance(table, dict) and _is_str(table.get("path")) and table["path"]):
                errors.append(f"{where}.{stage} must be a table with a path")
    versions = [v.get("version") for v in features.values() if isinstance(v, dict)]
    dupes = sorted({v for v in versions if v is not None and versions.count(v) > 1})
    if dupes:
        errors.append(f"[features]: version(s) {', '.join(map(str, dupes))} declared twice")


def _check_classes(config: dict, errors: list[str]) -> None:
    classes = _table(config, "classes", errors)
    errors += _unknown("[classes]", classes, {"control", "case"})
    for key in ("control", "case"):
        if key in classes and not (_is_str(classes[key]) and classes[key].strip()):
            errors.append(f"[classes].{key} must be a non-empty name")


RENAMED = {"tools": VIRUS_IDENTIFICATION_TOOLS_SECTION}  # earlier section names -> current ones


def check_config(config: dict) -> list[str]:
    """Every problem with a loaded configuration (empty: well formed)."""
    errors = [
        f"[{old}] is now called [{new}]: rename the section"
        for old, new in RENAMED.items()
        if old in config
    ]
    errors += _unknown("config", {k: v for k, v in config.items() if k not in RENAMED}, TOP_LEVEL)
    workspace = config.get("workspace")
    if workspace is not None and not (_is_str(workspace) and workspace):
        errors.append("workspace must be a folder (string)")
    for check in (
        _check_locations,
        _check_tools,
        _check_modalities,
        _check_workflow,
        _check_threads,
        _check_string_tables,
        _check_features,
        _check_classes,
    ):
        check(config, errors)
    return errors
