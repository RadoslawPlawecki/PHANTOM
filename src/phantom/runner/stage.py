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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from phantom.config.features import FeatureConfigManager
from phantom.config.loader import ConfigLoader


@dataclass
class RunContext:
    config: dict
    config_path: Path
    config_mgr: FeatureConfigManager

    @classmethod
    def load(cls, config_path: Path | None = None) -> RunContext:
        loader = ConfigLoader(config_path)
        config = loader.load()
        return cls(
            config=config, config_path=loader.path, config_mgr=FeatureConfigManager(loader.path)
        )

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
        """Checks that need config/data (versions exist, input files exist)."""
        return []

    @abstractmethod
    def inputs(self, p: Any, ctx: RunContext) -> list[Path]:
        """Files/directories whose content determines this job's result."""

    @abstractmethod
    def writes(self, p: Any, ctx: RunContext) -> list[Path]:
        """Directories this job writes into (used to propagate staleness)."""

    @abstractmethod
    def run(self, p: Any, ctx: RunContext) -> list[Path]:
        """Does the work; returns the output paths actually produced."""

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
