"""
Imports external feature sources (e.g. PhaBOX2 prediction folders) into the
``raw`` stage of a feature version, creating the version if asked to.
"""

import shutil
from pathlib import Path

from phantom.config.feature_versions import FeatureConfigManager


class FeatureImporter:
    def __init__(self, config_path: Path | str | None = None):
        self.config_mgr = FeatureConfigManager(config_path)

    def import_sources(
        self, sources: list[Path], version: str, create: bool, overwrite: bool = False
    ) -> tuple[Path, list[str]]:
        """
        Copies ``sources`` into the ``raw`` stage of ``version`` -- a new
        version (``create=True``, recorded in the configuration) or an
        existing one. A source already present is replaced only with
        ``overwrite``. Returns the raw directory and the sources that could
        not be copied.
        """
        existing = self.config_mgr.get_existing_versions()
        if create:
            if version in existing:
                raise ValueError(f"Version {version} already exists.")
            print(f"\n[INFO] Recording feature version {version} in the configuration...")
            target = self.config_mgr.create_new_version(version)
        else:
            if version not in existing:
                raise ValueError(
                    f"Version {version} does not exist (create it with create = true)."
                )
            target = self.config_mgr.get_stage_dir(version, "raw")
        return target, self._copy(sources, target, overwrite)

    @staticmethod
    def _copy(sources: list[Path], dest_dir: Path, overwrite: bool) -> list[str]:
        print(f"\n[INFO] Copying data to {dest_dir} ...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        failed = []
        for src in sources:
            target = dest_dir / src.name
            if target.exists():
                if not overwrite:
                    print(f"     Skipped (already there): {src.name}")
                    continue
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            try:
                if src.is_dir():
                    shutil.copytree(src, target)
                else:
                    shutil.copy2(src, target)
                print(f"     Copied: {src.name}")
            except OSError as e:
                print(f"[ERROR] Error copying {src.name}: {e}")
                failed.append(f"{src}: {e}")
        return failed
