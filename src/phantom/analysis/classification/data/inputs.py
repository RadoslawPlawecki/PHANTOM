"""
Locates classification inputs: the extracted feature matrix of a given
modality and virus identification tool.
"""

from pathlib import Path

from phantom.config.feature_versions import FeatureConfigManager


def feature_files(
    config_mgr: FeatureConfigManager, version: str, modality_cfg: dict
) -> tuple[Path, list[Path]]:
    """
    All extracted ``*_<tag>_FEAT.csv`` files of one modality (a
    ``[modalities.<key>]`` table: ``tool``, ``tag``) in a feature version,
    plus the directory they were searched in.
    """
    extracted_dir = config_mgr.get_stage_dir(version, "extracted") / modality_cfg["tool"]
    return extracted_dir, sorted(extracted_dir.glob(f"*_{modality_cfg['tag']}_FEAT.csv"))


def find_feature_file(
    config_mgr: FeatureConfigManager, version: str, modality_cfg: dict, vtool: str
) -> Path:
    """
    The extracted feature file of one modality for one virus identification
    tool (``<vtool>_..._<tag>_FEAT.csv``). Raises FileNotFoundError naming the
    searched directory if there is none.
    """
    extracted_dir, files = feature_files(config_mgr, version, modality_cfg)
    match = next((f for f in files if f.name.startswith(f"{vtool}_")), None)
    if match is None:
        raise FileNotFoundError(
            f"No {vtool} {modality_cfg['tag']} feature file (*_{modality_cfg['tag']}_FEAT.csv) "
            f"in {extracted_dir}"
        )
    return match
