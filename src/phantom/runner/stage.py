"""
The stage contract and registry.

Every pipeline step is a :class:`Stage` subclass registered with
:func:`register`. A stage declares its parameters (``Params``), what it reads
(``inputs``) and where it writes (``writes``), and does the actual work in
``run`` -- without ever prompting. Run files, ``phantom plan`` and stamps are
all built on these declarations.
"""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from phantom.config.features import FeatureConfigManager
from phantom.config.loader import ConfigLoader


@dataclass
class RunContext:
    config: dict
    config_path: Path
    config_mgr: FeatureConfigManager
    # Feature versions an earlier step of the current run file will create.
    planned_versions: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, config_path: Path | None = None) -> RunContext:
        loader = ConfigLoader(config_path)
        config = loader.load()
        return cls(
            config=config, config_path=loader.path, config_mgr=FeatureConfigManager(loader.path)
        )

    def reload(self) -> None:
        """Re-read config.toml after a stage changed it (e.g. a new feature version)."""
        self.config = ConfigLoader(self.config_path).load()

    def section_path(self, section: str) -> Path:
        """
        The directory a top-level ``config.toml`` section points to (e.g.
        ``[results]``), resolved against the data workspace. There is no
        fallback in code: a missing section is a configuration error.
        """
        rel = self.config.get(section, {}).get("path")
        if not rel:
            raise ValueError(f"config.toml is missing [{section}].path")
        return ConfigLoader.resolve_data_path(rel)

    def resolve(self, path: str | Path) -> Path:
        """A user-supplied path: absolute as is, relative to the data workspace."""
        path = Path(path)
        return path if path.is_absolute() else ConfigLoader.resolve_data_path(path)

    @property
    def vtools(self) -> list[str]:
        return list(self.config.get("tools", {}))

    @property
    def modalities(self) -> dict[str, dict]:
        return dict(self.config.get("modalities", {}))

    def feature_versions(self) -> list[str]:
        return self.config_mgr.get_existing_versions()

    def version_known(self, version: str) -> bool:
        return version in self.feature_versions() or version in self.planned_versions

    def stage_dir(self, version: str, stage: str) -> Path:
        """A feature version's stage directory -- from config.toml, or, for a
        version an earlier step creates, where it will be created."""
        if version in self.planned_versions and version not in self.feature_versions():
            return self.config_mgr.new_version_stage_dir(version, stage)
        return self.config_mgr.get_stage_dir(version, stage)


class Stage(ABC):
    name: ClassVar[str]
    summary: ClassVar[str]
    Params: ClassVar[type]

    def defaults(self, ctx: RunContext) -> dict[str, Any]:
        """Defaults that depend on config.toml (e.g. "all vtools")."""
        return {}

    def choices(self, ctx: RunContext) -> dict[str, list[Any]]:
        """Values available in config/data per field, shown in templates."""
        return {}

    def validate(self, p: Any, ctx: RunContext) -> list[str]:
        """Parameter/config checks (version declared, names valid) -- always
        run before anything executes."""
        return []

    def announces(self, p: Any) -> list[str]:
        """Feature versions this job creates (later steps may then use them)."""
        return []

    def check_inputs(self, p: Any, ctx: RunContext) -> list[str]:
        """
        Checks on the input *data* (files exist, columns present). Run before
        anything executes for jobs whose inputs already exist, and right
        before the job itself for jobs whose inputs an earlier job of the
        same run produces (e.g. extract after merge in one run file).
        """
        return []

    @abstractmethod
    def inputs(self, p: Any, ctx: RunContext) -> list[Path]:
        """
        Files/directories whose content determines this job's result. Must
        be computable before the inputs exist (prefer stable directories over
        discovered file lists), so dependencies between jobs can be planned.
        """

    @abstractmethod
    def writes(self, p: Any, ctx: RunContext) -> list[Path]:
        """Directories this job writes into (used to propagate staleness)."""

    @abstractmethod
    def run(self, p: Any, ctx: RunContext) -> list[Path] | None:
        """Does the work; returns the output paths actually produced, or None
        when nothing was produced on purpose (a dry run) -- then no stamp is
        written, so the job still counts as not done."""

    def label(self, p: Any) -> str:
        """Short one-line description of a job, for ``phantom plan``."""
        return ""


REGISTRY: dict[str, Stage] = {}


def register(cls: type[Stage]) -> type[Stage]:
    if cls.name in REGISTRY:
        raise ValueError(f"Stage {cls.name!r} registered twice")
    REGISTRY[cls.name] = cls()
    return cls


def get_stage(name: str) -> Stage:
    if name not in REGISTRY:
        hint = difflib.get_close_matches(name, REGISTRY, n=1)
        suffix = f" -- did you mean {hint[0]!r}?" if hint else ""
        available = ", ".join(sorted(REGISTRY)) or "none"
        raise KeyError(f"Unknown stage {name!r}{suffix} (available: {available})")
    return REGISTRY[name]
