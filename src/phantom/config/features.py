"""
Feature versions in the configuration: listing them, resolving their stage
directories, recording new ones.
"""

from pathlib import Path

import tomlkit

from phantom.config.loader import ConfigLoader

# Stage sub-tables written for every new feature version: key -> (folder, description).
NEW_VERSION_STAGES = {
    "raw": (
        "raw",
        "Raw feature files stored separately for each sample before merging and preprocessing.",
    ),
    "raw_merged": (
        "raw-merged",
        "Merged raw feature data combined into unified files containing all samples.",
    ),
    "preprocessed": ("preprocessed", "Preprocessed feature data."),
    "extracted": (
        "extracted",
        "Extracted feature representations used as input for classifier training and prediction.",
    ),
}


class FeatureConfigManager:
    """
    Feature versions: reads them from the merged configuration and records
    new ones in ``config.local.toml`` -- they describe this dataset, not the
    code, so they never go into the tracked config.toml.
    """

    def __init__(self, config_path: Path | str | None = None):
        self.loader = ConfigLoader(Path(config_path) if config_path else None)
        self.config_path = self.loader.path

    def _features(self) -> dict:
        return self.loader.load().get("features", {})

    def _features_root(self) -> str:
        root = self._features().get("path")
        if not root:
            raise ValueError("config.toml is missing [features].path")
        return root

    def get_existing_versions(self) -> list[str]:
        return [
            v["version"]
            for v in self._features().values()
            if isinstance(v, dict) and "version" in v
        ]

    def create_new_version(self, version_str: str) -> Path:
        """Records version ``version_str`` (stage layout NEW_VERSION_STAGES) in
        config.local.toml; returns its raw-stage directory."""
        if version_str in self.get_existing_versions():
            raise ValueError(f"Version {version_str} already exists in config.")
        base_path = f"{self._features_root()}/{version_str}"
        local = self.loader.local_path
        doc = (
            tomlkit.parse(local.read_text(encoding="utf-8"))
            if local.is_file()
            else tomlkit.document()
        )
        if "features" not in doc:
            doc.add("features", tomlkit.table(is_super_table=True))
        v_table = tomlkit.table()
        v_table.add("path", base_path)
        v_table.add("version", version_str)
        v_table.add("type", "directory")
        for sub_key, (folder_name, desc) in NEW_VERSION_STAGES.items():
            sub_table = tomlkit.table()
            sub_table.add("path", folder_name)
            sub_table.add("type", "directory")
            sub_table.add("description", desc)
            v_table.add(sub_key, sub_table)
        doc["features"].add(f"v{version_str.replace('.', '_')}", v_table)
        local.write_text(tomlkit.dumps(doc), encoding="utf-8")
        return ConfigLoader.resolve_data_path(base_path) / NEW_VERSION_STAGES["raw"][0]

    def new_version_stage_dir(self, version_str: str, stage: str) -> Path:
        """
        Where ``create_new_version(version_str)`` *will* put a stage -- for
        planning jobs on a version a run file creates before using it.
        """
        folder = NEW_VERSION_STAGES[stage][0]
        return ConfigLoader.resolve_data_path(f"{self._features_root()}/{version_str}") / folder

    def get_version_path(self, version_str: str) -> Path:
        return ConfigLoader.resolve_data_path(self._get_version_table(version_str)["path"])

    def get_stage_dir(self, version_str: str, stage: str) -> Path:
        """
        Resolves a version's stage directory (e.g. "raw", "raw_merged",
        "preprocessed", "extracted") from its configuration entry, rather
        than assuming a hardcoded folder name.
        """
        version_table = self._get_version_table(version_str)
        if stage not in version_table:
            raise ValueError(f"Stage '{stage}' not found for version {version_str} in config.")
        return self.get_version_path(version_str) / version_table[stage]["path"]

    def _get_version_table(self, version_str: str) -> dict:
        v_key = f"v{version_str.replace('.', '_')}"
        features = self._features()
        if v_key not in features:
            raise ValueError(f"Version {version_str} not found in config.")
        return features[v_key]
