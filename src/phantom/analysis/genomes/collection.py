"""
Copies the putative viral genomes (*.fna) each virus identification tool
found in a sample's assembly into the genome registry, as
``<[genomes]>/<tool>/S<n>_<abbr>.fna``.
"""

import re
import shutil

import pandas as pd

from phantom.config.loader import ConfigLoader


class GenomeCollector:
    def __init__(self, config: dict, metadata_table):
        self.config = config
        self.tools_map = ConfigLoader.virus_identification_tools(self.config)
        self.base_dest = ConfigLoader.resolve_data_path(self.config["genomes"]["path"])
        self.metadata_table = metadata_table

    def _is_valid_format(self, folder_name: str, tool_abbreviation: str) -> bool:
        pattern = rf"^S\d+_{re.escape(tool_abbreviation)}$"
        return bool(re.match(pattern, folder_name))

    def scan_for_invalid(self) -> int:
        invalid_count = 0
        for tool_abbreviation, tool_name in self.tools_map.items():
            tool_config = self.config.get(tool_name)
            if not tool_config:
                continue
            root = ConfigLoader.resolve_data_path(tool_config["path"])
            for file in root.rglob(tool_config["viral_fna"]):
                parts = file.relative_to(root).parts
                subfolder = parts[0] if len(parts) > 1 else "root"
                if not self._is_valid_format(subfolder, tool_abbreviation):
                    invalid_count += 1
        return invalid_count

    def collect(self) -> pd.DataFrame:
        """
        Collects files and returns a wide-format DataFrame joined on sample ID.
        """
        all_samples = {}
        for tool_abbreviation, tool_name in self.tools_map.items():
            tool_config = self.config.get(tool_name)
            if not tool_config:
                continue
            root = ConfigLoader.resolve_data_path(tool_config["path"])
            dest_dir = self.base_dest / tool_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file in root.rglob(tool_config["viral_fna"]):
                parts = file.relative_to(root).parts
                subfolder = parts[0] if len(parts) > 1 else "root"
                if not self._is_valid_format(subfolder, tool_abbreviation):
                    continue  # ids come only from preprocessing.samples
                sample_id = subfolder.split("_")[0]
                formatted_name = f"{subfolder}.fna"
                dst_file = dest_dir / formatted_name
                shutil.copy2(file, dst_file)
                if sample_id not in all_samples:
                    all_samples[sample_id] = {"id": sample_id}
                sample = all_samples[sample_id]
                sample[f"{tool_abbreviation}_original_file"] = file.name
                sample[f"{tool_abbreviation}_formatted_file"] = formatted_name
        return pd.DataFrame(list(all_samples.values()))
