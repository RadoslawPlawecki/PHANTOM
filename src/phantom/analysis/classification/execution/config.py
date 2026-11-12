"""
Run configurations of the classification executors (built by the
classification.* stages in phantom.stages.classification).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MdsConfig:
    comp: str
    host: str
    func: str
    out_dir: str


@dataclass(frozen=True)
class SingleModalConfig:
    in_file: str
    out_dir: str
    modality: str | None
    vtool: str | None
    run_loocv: bool
    run_repeated: bool
    use_smote: bool
    model_type: str


@dataclass(frozen=True)
class MultiModalConfig:
    comp: str
    host: str
    func: str
    out_dir: str | None
    run_loocv: bool
    run_repeated: bool
    use_smote: bool
    model_type: str
    fusion: str
    opt: str | None
    n_trials: int | None
