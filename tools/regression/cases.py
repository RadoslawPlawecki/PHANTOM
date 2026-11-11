"""
Regression cases: each pipeline action, answered the same way on the legacy
(menu) path and the new (run-file) path. Choices deliberately exercise the
non-default branches (a CheckV mask, count matrices, a non-default ratio
mode, both validators, ...), so a dropped or mis-mapped option shows up as a
changed output rather than passing silently.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from harness import SANDBOX, Case, Patcher

V = "3.0"
FEAT = f"data/features/{V}"
META = "data/genomes/genome_metadata.csv"
MASK = "MQ_vMAGs.tsv"
VTOOL = "geN"
EXTRACT_CHOICES = {
    "cherry": {"feature_col": "ncbi_genus", "binary": True, "min_patients": 2},
    "phagcn": {"feature_col": "genus", "binary": False, "min_patients": 3},
}


def results(*parts: str) -> Path:
    return SANDBOX.joinpath("results", *parts)


# --- features ----------------------------------------------------------------


def legacy_merge(p: Patcher) -> None:
    from phantom.cli.features import FeatureCollectionPrompts
    from phantom.features.controller import FeatureController

    p.answers(
        FeatureCollectionPrompts,
        ask_existing_version=V,
        ask_id_column=lambda label, cols: "Genome" if label.startswith("phavip") else "Accession",
    )
    FeatureController()._merge_features()


def legacy_preprocess(mask: str | None):
    def run(p: Patcher) -> None:
        from phantom.cli.features import FeatureCollectionPrompts, FeatureExtractionPrompts
        from phantom.features.controller import FeatureController

        p.answers(FeatureCollectionPrompts, ask_existing_version=V)
        p.set(
            FeatureExtractionPrompts,
            "ask_mask_file",
            lambda self, in_root: (self._get_tool_mask_dir(in_root) / mask) if mask else None,
        )
        FeatureController()._preprocess_features()

    return run


def legacy_extract(p: Patcher) -> None:
    from phantom.cli.features import FeatureCollectionPrompts, FeatureExtractionPrompts
    from phantom.features.controller import FeatureController

    # The legacy prompts don't say which tool they are asked for; the column
    # prompt sees the dataframe, so it records the tool for the next two.
    current = {}

    def ask_column(self, df):
        tool = "cherry" if "ncbi_genus" in df.columns else "phagcn"
        current["tool"] = tool
        return EXTRACT_CHOICES[tool]["feature_col"]

    p.answers(FeatureCollectionPrompts, ask_existing_version=V)
    p.set(FeatureExtractionPrompts, "ask_column", ask_column)
    p.set(
        FeatureExtractionPrompts,
        "ask_binary",
        lambda self, default=True: EXTRACT_CHOICES[current["tool"]]["binary"],
    )
    p.set(
        FeatureExtractionPrompts,
        "ask_min_patients",
        lambda self, default=1, max_value=10: EXTRACT_CHOICES[current["tool"]]["min_patients"],
    )
    FeatureController()._extract_features()


def legacy_diversity(ratio_mode: str):
    def run(p: Patcher) -> None:
        from phantom.cli.features import FeatureCollectionPrompts, FeatureDiversityPrompts
        from phantom.features.controller import FeatureController

        p.answers(FeatureCollectionPrompts, ask_existing_version=V)
        p.answers(
            FeatureDiversityPrompts,
            ask_out_dir=results("diversity", V),
            ask_ratio_mode=ratio_mode,
        )
        FeatureController()._analyze_diversity()

    return run


def legacy_summarize(p: Patcher) -> None:
    from phantom.cli.features import FeatureCollectionPrompts
    from phantom.features.controller import FeatureController

    p.answers(FeatureCollectionPrompts, ask_existing_version=V)
    FeatureController()._summarize_functions()


def legacy_import(p: Patcher) -> None:
    from phantom.cli.features import FeatureCollectionPrompts
    from phantom.features.controller import FeatureController

    p.answers(
        FeatureCollectionPrompts,
        ask_feature_paths=[SANDBOX / "data/phabox2/cherry/genomad"],
        ask_version_strategy="Create a new feature version",
        ask_new_version="8.0",
        ask_auto_merge=False,
    )
    FeatureController()._import_features()


OPTIMIZE_KNOBS = {
    "model": "rf",
    "validator": "loocv",
    "metric": "mcc",
    "smote": False,
    "outer_folds": 2,
    "outer_repeats": 1,
    "inner_trials": 2,
    "top_k_features": 5,
    "n_permutations": 2,
    "permutation_trials": 2,
}


def legacy_optimize(tool: str, mode: str):
    def run(p: Patcher) -> None:
        from phantom.cli.features import (
            FeatureCollectionPrompts,
            FeatureExtractionPrompts,
            FeatureOptimizationPrompts,
        )
        from phantom.features.optimization.optimizer import FeatureOptimizer

        k = OPTIMIZE_KNOBS
        p.answers(FeatureCollectionPrompts, ask_existing_version=V)
        p.answers(
            FeatureOptimizationPrompts,
            ask_tool_choice=tool,
            ask_mode_choice=mode,
            ask_model_choice=k["model"],
            ask_validator_choice=k["validator"],
            ask_metric_choice=k["metric"],
            ask_smote_choice=k["smote"],
            ask_outer_folds=k["outer_folds"],
            ask_outer_repeats=k["outer_repeats"],
            ask_inner_trials=k["inner_trials"],
            ask_top_k_features=k["top_k_features"],
            ask_n_permutations=k["n_permutations"],
            ask_permutation_trials=k["permutation_trials"],
            ask_optimization_out_dir=results("optimization", tool),
            ask_file_choice=lambda label, files: next(
                f for f in files if f.name.startswith(f"{VTOOL}_")
            ),
        )
        p.set(
            FeatureExtractionPrompts,
            "ask_mask_file",
            lambda self, in_root: self._get_tool_mask_dir(in_root) / MASK,
        )
        FeatureOptimizer().run()

    return run


# --- classification ----------------------------------------------------------


def legacy_single(p: Patcher) -> None:
    from phantom.classification.controller import ClassificationController
    from phantom.cli.classification import ClassificationPrompts

    p.answers(
        ClassificationPrompts,
        ask_feature_version=V,
        ask_grid_scope="Single file",  # the string the controller compares against
        ask_vtool_choice=VTOOL,
        ask_omic_choice="host",
        ask_model_choice="rf",
        ask_validators=(True, True),
        ask_smote_choice=False,
        ask_classification_out_dir=results("single-omic", V),
    )
    ClassificationController()._run_single_omic()


def legacy_multi(fusion: str):
    def run(p: Patcher) -> None:
        from phantom.classification.controller import ClassificationController
        from phantom.cli.classification import ClassificationPrompts

        p.answers(
            ClassificationPrompts,
            ask_feature_version=V,
            ask_grid_scope="Single file",
            ask_vtool_choice=VTOOL,
            ask_model_choice="rf",
            ask_validators=(True, False),
            ask_smote_choice=False,
            ask_fusion_choice=fusion,
            ask_late_fusion_opt=(True, 3),
            ask_classification_out_dir=results("multi-omic", V),
        )
        ClassificationController()._run_multi_omic()

    return run


def legacy_mds(p: Patcher) -> None:
    from phantom.classification.controller import ClassificationController
    from phantom.cli.classification import ClassificationPrompts

    p.answers(
        ClassificationPrompts,
        ask_feature_version=V,
        ask_grid_scope="Single file",
        ask_vtool_choice=VTOOL,
        ask_classification_out_dir=results("mds", V),
    )
    ClassificationController()._run_unsupervised()


# --- preprocessing -----------------------------------------------------------


def legacy_collect(p: Patcher) -> None:
    from phantom.cli.preprocessing import PreprocessingPrompts
    from phantom.preprocessing.controller import PreprocessingController

    p.answers(
        PreprocessingPrompts,
        ask_auto_format=False,
        ask_metadata_targets=["1) Renaming (format file paths)"],
    )
    ctrl = PreprocessingController()
    df = ctrl._collect_genomes()
    ctrl._gather_metadata(df)


def prepare_renaming_only_metadata(root: Path) -> None:
    """Metadata as it is right after collection: id + renaming columns."""
    path = root / META
    df = pd.read_csv(path, sep=";")
    keep = ["id"] + [c for c in df.columns if c.endswith(("_original_file", "_formatted_file"))]
    df[keep].to_csv(path, sep=";", index=False)


def legacy_metadata(p: Patcher) -> None:
    from phantom.cli.preprocessing import PreprocessingPrompts
    from phantom.preprocessing.controller import PreprocessingController

    p.answers(
        PreprocessingPrompts,
        ask_metadata_targets=[
            "2) Labeling (assign labels to samples)",
            "3) Runtimes",
            "4) CheckV stats",
            "5) Input file sizes",
        ],
        ask_strategy="2) Sample ID Threshold (e.g., S <= n)",
        ask_threshold=34,
    )
    PreprocessingController()._gather_metadata()


def legacy_masks(p: Patcher) -> None:
    from phantom.preprocessing.controller import PreprocessingController

    PreprocessingController()._generate_checkv_masks()


def legacy_visualize(p: Patcher) -> None:
    from phantom.cli.preprocessing import PreprocessingPrompts
    from phantom.preprocessing.controller import PreprocessingController

    p.answers(
        PreprocessingPrompts,
        ask_plots=lambda available: list(available),
        ask_plot_out_dir=results("plots", "metadata"),
    )
    PreprocessingController()._visualize_metadata()


TOOL_INPUTS = ["data/genomad", "data/vibrant", "data/virsorter2"]

CASES = [
    Case(
        "features.merge",
        outputs=[f"{FEAT}/raw-merged"],
        legacy=legacy_merge,
        steps=[
            {
                "stage": "features.merge",
                "version": V,
                "id_columns": {"phavip/*": "Genome", "*": "Accession"},
            }
        ],
        links=[f"{FEAT}/raw"],
        real={f"{FEAT}/raw-merged": f"{FEAT}/raw-merged"},
    ),
    Case(
        "features.preprocess",
        outputs=[f"{FEAT}/preprocessed"],
        legacy=legacy_preprocess(MASK),
        steps=[{"stage": "features.preprocess", "version": V, "mask": MASK}],
        links=[f"{FEAT}/raw-merged", "data/masks"],
        real={f"{FEAT}/preprocessed": f"{FEAT}/preprocessed"},
    ),
    Case(
        "features.preprocess.nomask",
        outputs=[f"{FEAT}/preprocessed"],
        legacy=legacy_preprocess(None),
        steps=[{"stage": "features.preprocess", "version": V}],
        links=[f"{FEAT}/raw-merged", "data/masks"],
        real={f"{FEAT}/preprocessed": f"{FEAT}/preprocessed"},
    ),
    Case(
        "features.extract",
        outputs=[f"{FEAT}/extracted"],
        legacy=legacy_extract,
        steps=[
            {
                "stage": "features.extract",
                "version": V,
                "feature_col": {t: c["feature_col"] for t, c in EXTRACT_CHOICES.items()},
                "binary": {t: c["binary"] for t, c in EXTRACT_CHOICES.items()},
                "min_patients": {t: c["min_patients"] for t, c in EXTRACT_CHOICES.items()},
            }
        ],
        links=[f"{FEAT}/preprocessed"],
        real={f"{FEAT}/extracted": f"{FEAT}/extracted"},
    ),
    *[
        Case(
            f"features.diversity.{mode}",
            outputs=["results/diversity"],
            legacy=legacy_diversity(mode),
            steps=[{"stage": "features.diversity", "version": V, "ratio_mode": mode}],
            links=[f"{FEAT}/preprocessed"],
            copies=[META],
        )
        for mode in ("ratio_of_means", "mean_of_ratios")
    ],
    Case(
        "features.summarize_functions",
        outputs=["data/functions"],
        legacy=legacy_summarize,
        steps=[{"stage": "features.summarize_functions", "version": V}],
        links=["data/phanotate", f"{FEAT}/preprocessed"],
        keep_limit_mb=0,
    ),
    Case(
        "features.import",
        outputs=["data/features/8.0", "config.toml"],
        legacy=legacy_import,
        steps=[
            {
                "stage": "features.import",
                "version": "8.0",
                "create": True,
                "sources": ["data/phabox2/cherry/genomad"],
            }
        ],
        links=["data/phabox2"],
    ),
    *[
        Case(
            f"features.optimize.{tool}.{mode}",
            outputs=["results/optimization"],
            legacy=legacy_optimize(tool, mode),
            steps=[
                {
                    "stage": "features.optimize",
                    "version": V,
                    "tool": tool,
                    "vtool": VTOOL,
                    "mode": mode,
                    "mask": MASK,
                    # the legacy run's location (the new default adds version/vtool)
                    "out_dir": f"results/optimization/{tool}",
                    **OPTIMIZE_KNOBS,
                }
            ],
            links=[f"{FEAT}/raw-merged", "data/masks"],
            copies=[META],
        )
        for tool, mode in (("phagcn", "all"), ("cherry", "exploratory"), ("phavip", "exploratory"))
    ],
    Case(
        "classification.single",
        outputs=["results/single-omic"],
        legacy=legacy_single,
        steps=[
            {
                "stage": "classification.single",
                "version": V,
                "modality": "host",
                "vtool": VTOOL,
                "model": "rf",
                "validators": ["loocv", "rcv"],
            }
        ],
        links=[f"{FEAT}/extracted"],
        copies=[META],
    ),
    *[
        Case(
            f"classification.multi.{fusion}",
            outputs=["results/multi-omic"],
            legacy=legacy_multi(fusion),
            steps=[
                {
                    "stage": "classification.multi",
                    "version": V,
                    "vtool": VTOOL,
                    "fusion": fusion,
                    "model": "rf",
                    "validators": ["loocv"],
                    **({"optimize_weights": True, "n_trials": 3} if fusion == "late" else {}),
                }
            ],
            links=[f"{FEAT}/extracted"],
            copies=[META],
        )
        for fusion in ("early", "late")
    ],
    Case(
        "classification.mds",
        outputs=["results/mds"],
        legacy=legacy_mds,
        steps=[{"stage": "classification.mds", "version": V, "vtool": VTOOL}],
        links=[f"{FEAT}/extracted"],
        copies=[META],
    ),
    Case(
        "preprocessing.collect",
        outputs=["data/genomes"],
        legacy=legacy_collect,
        steps=[{"stage": "preprocessing.collect"}],
        links=TOOL_INPUTS,
        keep_limit_mb=0,
    ),
    Case(
        "preprocessing.metadata",
        outputs=[META],
        legacy=legacy_metadata,
        steps=[
            {
                "stage": "preprocessing.metadata",
                "targets": ["labeling", "runtimes", "checkv", "sizes"],
                "labeling": {"strategy": "threshold", "threshold": 34},
            }
        ],
        links=[*TOOL_INPUTS, "data/checkv", "data/megahit"],
        copies=[META],
        prepare=prepare_renaming_only_metadata,
        real={META: META},
    ),
    Case(
        "preprocessing.masks",
        outputs=["data/masks"],
        legacy=legacy_masks,
        steps=[{"stage": "preprocessing.masks"}],
        links=["data/checkv"],
        real={"data/masks": "data/masks"},
    ),
    Case(
        "preprocessing.visualize",
        outputs=["results/plots"],
        legacy=legacy_visualize,
        steps=[{"stage": "preprocessing.visualize"}],
        copies=[META],
    ),
]

BY_NAME = {c.name: c for c in CASES}
