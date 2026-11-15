"""
Locates and loads the project's configuration.

The configuration is ``config.toml`` (in the repository, generic: works on any
machine) merged with an optional ``config.local.toml`` next to it (not
tracked: this machine's tool environments, core count, and the feature
versions of this dataset). Tables merge key by key; everything else in the
local file replaces the base value.

The configuration a ``phantom`` command was started with (``--config``) is
*activated* for the whole process, so code that has no configuration passed
to it (e.g. the class labels) reads the same one. An optional top-level
``workspace = "<dir>"`` (relative to the configuration file) moves the data
workspace; without it the workspace is the repository's parent folder.
"""

import tomllib
from pathlib import Path
from typing import Any

LOCAL_NAME = "config.local.toml"
# The virus identification tools: abbreviation (used in file names and sample
# ids, e.g. "geN|S1") -> tool name (e.g. "genomad").
VIRUS_IDENTIFICATION_TOOLS_SECTION = "virus-identification-tools"


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """``override`` on top of ``base``: nested tables merged, other values replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class ConfigLoader:
    # Root of this codebase (where config.toml itself lives).
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Root for the data workspace that config.toml's paths describe.
    # That workspace is a sibling of the project, not part of it: the
    # pipeline reads/writes genomes, features, results, etc. next to the
    # codebase, never inside it. Resolve every path taken from config.toml
    # through `resolve_data_path()` below rather than joining it onto
    # PROJECT_ROOT directly.
    DATA_ROOT = PROJECT_ROOT.parent

    DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"

    # The configuration of this process, set by activate() (None: the default).
    ACTIVE_PATH: Path | None = None

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else (self.ACTIVE_PATH or self.DEFAULT_CONFIG_PATH)
        self.config: dict | None = None

    @classmethod
    def activate(cls, path: Path | None = None) -> "ConfigLoader":
        """
        Makes ``path`` (default: the repository's config.toml) the
        configuration of this process and applies its ``workspace`` key.
        """
        from phantom.config.schema import ConfigError, check_config

        loader = cls(Path(path) if path else cls.DEFAULT_CONFIG_PATH)
        config = loader.load()
        errors = check_config(config)
        if errors:
            raise ConfigError([f"{loader.path}: {e}" for e in errors])
        cls.ACTIVE_PATH = loader.path.resolve()
        workspace = config.get("workspace")
        if workspace is not None:
            if not isinstance(workspace, str) or not workspace:
                raise ValueError(f"{loader.path}: workspace must be a folder name or path")
            cls.DATA_ROOT = (loader.path.parent / workspace).resolve()
        return loader

    @property
    def local_path(self) -> Path:
        return self.path.with_name(LOCAL_NAME)

    @classmethod
    def resolve_data_path(cls, relative_path: str | Path) -> Path:
        """
        Resolves a path declared in config.toml (e.g. "data/genomes")
        against DATA_ROOT, the data workspace sitting next to the project.
        """
        return cls.DATA_ROOT / relative_path

    @staticmethod
    def virus_identification_tools(config: dict) -> dict[str, str]:
        """``[virus-identification-tools]``: abbreviation -> tool name, in config order."""
        return dict(config.get(VIRUS_IDENTIFICATION_TOOLS_SECTION, {}))

    @staticmethod
    def virus_identification_tool_abbreviation(config: dict, name: str) -> str | None:
        """The abbreviation of a virus identification tool given by its
        abbreviation ("VS2") or its full name ("virsorter2", any case); None if unknown."""
        tools = ConfigLoader.virus_identification_tools(config)
        if name in tools:
            return name
        by_name = {full.lower(): abbreviation for abbreviation, full in tools.items()}
        return by_name.get(name.lower())

    @classmethod
    def config_path(cls, config: dict, section: str, key: str = "path") -> Path:
        """
        A location from a loaded config (``[section].key``), resolved against
        the data workspace. There are deliberately no defaults in code: a
        missing entry is a configuration error naming what to add.
        """
        value = config.get(section, {}).get(key)
        if not value:
            raise ValueError(f"config.toml is missing [{section}].{key}")
        return cls.resolve_data_path(value)

    def load(self) -> dict:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {self.path} (it ships with the repository; "
                "pass --config to use another one)"
            )
        with open(self.path, "rb") as f:
            config = tomllib.load(f)
        if self.local_path.is_file():
            with open(self.local_path, "rb") as f:
                config = merge(config, tomllib.load(f))
        self.config = config
        return config
