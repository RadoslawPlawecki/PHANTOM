"""
Coordinates importing feature sources into the project structure.

:meth:`FeatureImporter.import_sources` does the work from explicit arguments
(run files); :meth:`FeatureImporter.run` is the legacy interactive menu that
asks for them first.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

from phantom.config.features import FeatureConfigManager

OverwriteChooser = Callable[[Path, Path], bool]


def never_overwrite(src: Path, target: Path) -> bool:
    return False


def always_overwrite(src: Path, target: Path) -> bool:
    return True


class FeatureImporter:
    def __init__(self, config_path: Path | str | None = None):
        self.config_mgr = FeatureConfigManager(config_path)

    def import_sources(
        self,
        sources: list[Path],
        version: str,
        create: bool,
        overwrite: OverwriteChooser = never_overwrite,
    ) -> tuple[Path, list[str]]:
        """
        Copies ``sources`` into the ``raw`` stage of ``version`` -- a new
        version (``create=True``, added to config.toml) or an existing one.
        Returns the raw directory and the sources that could not be copied.
        """
        existing = self.config_mgr.get_existing_versions()
        if create:
            if version in existing:
                raise ValueError(f"Version {version} already exists.")
            print(f"\n[INFO] Updating config.toml with v{version} schema...")
            target_raw_path = self.config_mgr.create_new_version(version)
        else:
            if version not in existing:
                raise ValueError(
                    f"Version {version} does not exist (create it with create = true)."
                )
            target_raw_path = self.config_mgr.get_stage_dir(version, "raw")
        failed = self._copy_features(sources, target_raw_path, overwrite)
        return target_raw_path, failed

    def run(self) -> Path | None:
        """Legacy interactive menu."""
        import questionary

        from phantom.cli.features import FeatureCollectionPrompts

        print("\n--- Feature Import & Versioning ---")
        source_paths = FeatureCollectionPrompts.ask_feature_paths()
        if not source_paths:
            print("[INFO] No paths provided. Aborting.")
            return None
        existing_versions = self.config_mgr.get_existing_versions()
        strategy = FeatureCollectionPrompts.ask_version_strategy(existing_versions)
        if strategy == "Cancel" or not strategy:
            return None
        if strategy == "Create a new feature version":
            version_str = FeatureCollectionPrompts.ask_new_version()
            create = True
        elif strategy == "Append to an existing version":
            version_str = FeatureCollectionPrompts.ask_existing_version(existing_versions)
            create = False
        else:
            return None
        if not version_str:
            return None

        def confirm(src: Path, target: Path) -> bool:
            return bool(
                questionary.confirm(
                    f"[WARNING] {src.name} already exists in {target.parent.name}. Overwrite?"
                ).ask()
            )

        try:
            target_raw_path, _ = self.import_sources(source_paths, version_str, create, confirm)
        except ValueError as e:
            print(f"[ERROR] {e} Aborting.")
            return None
        return target_raw_path

    def _copy_features(
        self, sources: list[Path], dest_dir: Path, overwrite: OverwriteChooser
    ) -> list[str]:
        print(f"\n[INFO] Copying data to {dest_dir} ...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        failed = []
        for src in sources:
            target_path = dest_dir / src.name
            if target_path.exists():
                if not overwrite(src, target_path):
                    print(f"     Skipped: {src.name}")
                    continue
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            try:
                if src.is_dir():
                    shutil.copytree(src, target_path)
                else:
                    shutil.copy2(src, target_path)
                success_count += 1
                print(f"     Copied: {src.name}")
            except Exception as e:
                print(f"[ERROR] Error copying {src.name}: {e}")
                failed.append(f"{src}: {e}")
        print(f"\n[SUCCESS] Imported {success_count} source(s) into {dest_dir}.")
        return failed
