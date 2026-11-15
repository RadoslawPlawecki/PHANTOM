"""
Regression cases: each pipeline action as a run-file step with fixed
parameters. Most baselines were recorded from the former interactive menu,
answered the same way, so a case passing means the stage reproduces the
menu's output; cases whose results changed on purpose since (seeded
randomness, version 1.0) were re-recorded, and classification.figures is
new. The choices deliberately exercise non-default branches (a
CheckV mask, count matrices, a non-default ratio mode, both validators, ...),
so a dropped or mis-mapped option shows up as a changed output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from harness import Case

V = "3.0"
FEAT = f"data/features/{V}"
META = "data/genomes/genome_metadata.csv"
MASK = "MQ_vMAGs.tsv"
VTOOL = "geN"
EXTRACT_CHOICES = {
    "cherry": {"feature_col": "ncbi_genus", "binary": True, "min_patients": 2},
    "phagcn": {"feature_col": "genus", "binary": False, "min_patients": 3},
}

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


def prepare_renaming_only_metadata(root: Path) -> None:
    """Metadata as it is right after collection: id + renaming columns."""
    path = root / META
    df = pd.read_csv(path, sep=";")
    keep = ["id"] + [c for c in df.columns if c.endswith(("_original_file", "_formatted_file"))]
    df[keep].to_csv(path, sep=";", index=False)


TOOL_INPUTS = ["data/genomad", "data/vibrant", "data/virsorter2"]

CASES = [
    Case(
        "features.merge",
        outputs=[f"{FEAT}/raw-merged"],
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
        steps=[{"stage": "features.preprocess", "version": V, "mask": MASK}],
        links=[f"{FEAT}/raw-merged", "data/masks"],
        real={f"{FEAT}/preprocessed": f"{FEAT}/preprocessed"},
    ),
    Case(
        "features.preprocess.nomask",
        outputs=[f"{FEAT}/preprocessed"],
        steps=[{"stage": "features.preprocess", "version": V}],
        links=[f"{FEAT}/raw-merged", "data/masks"],
        real={f"{FEAT}/preprocessed": f"{FEAT}/preprocessed"},
    ),
    Case(
        "features.extract",
        outputs=[f"{FEAT}/extracted"],
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
            steps=[{"stage": "features.diversity", "version": V, "ratio_mode": mode}],
            links=[f"{FEAT}/preprocessed"],
            copies=[META],
        )
        for mode in ("ratio_of_means", "mean_of_ratios")
    ],
    Case(
        "features.summarize_functions",
        outputs=["data/functions"],
        steps=[{"stage": "features.summarize_functions", "version": V}],
        links=["data/phanotate", f"{FEAT}/preprocessed"],
        keep_limit_mb=0,
    ),
    Case(
        "features.import",
        outputs=["data/features/8.0", "config.toml"],
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
            steps=[
                {
                    "stage": "features.optimize",
                    "version": V,
                    "tool": tool,
                    "vtool": VTOOL,
                    "mode": mode,
                    "mask": MASK,
                    # where the baseline was recorded (the default adds version/vtool)
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
        outputs=["results/single-modal"],
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
    Case(
        "classification.single.models",  # the algorithms added in 1.0
        outputs=["results/single-modal"],
        steps=[
            {
                "stage": "classification.single",
                "version": V,
                "modality": "host",
                "vtool": VTOOL,
                "model": ["logreg", "hgb", "xgb"],
                "validators": ["loocv"],
            }
        ],
        links=[f"{FEAT}/extracted"],
        copies=[META],
    ),
    *[
        Case(
            f"classification.multi.{fusion}",
            outputs=["results/multi-modal"],
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
        "classification.figures",
        outputs=["results/figures"],
        steps=[
            {
                "stage": "classification.single",
                "version": V,
                "modality": ["comp", "host"],
                "vtool": VTOOL,
                "model": "catboost",
                "validators": ["loocv"],
            },
            {
                "stage": "classification.figures",
                "version": V,
                "model": "catboost",
                "validators": ["loocv"],
                "kinds": ["single"],
            },
        ],
        links=[f"{FEAT}/extracted"],
        copies=[META],
    ),
    Case(
        "classification.mds",
        outputs=["results/mds"],
        steps=[{"stage": "classification.mds", "version": V, "vtool": VTOOL}],
        links=[f"{FEAT}/extracted"],
        copies=[META],
    ),
    Case(
        "preprocessing.collect",
        outputs=["data/genomes"],
        steps=[{"stage": "preprocessing.collect"}],
        links=TOOL_INPUTS,
        keep_limit_mb=0,
    ),
    Case(
        "preprocessing.metadata",
        outputs=[META],
        steps=[
            {
                "stage": "preprocessing.metadata",
                "targets": ["labeling", "runtimes", "checkv", "sizes"],
                "labeling": {"strategy": "threshold", "threshold": 34},
            }
        ],
        links=[*TOOL_INPUTS, "data/checkv", "data/masks", "data/assemblies"],
        copies=[META],
        prepare=prepare_renaming_only_metadata,
        real={META: META},
    ),
    Case(
        "preprocessing.masks",
        outputs=["data/masks"],
        steps=[{"stage": "preprocessing.masks"}],
        links=["data/checkv"],
        real={"data/masks": "data/masks"},
    ),
    Case(
        "preprocessing.visualize",
        outputs=["results/plots"],
        steps=[{"stage": "preprocessing.visualize"}],
        copies=[META],
    ),
]

BY_NAME = {c.name: c for c in CASES}
