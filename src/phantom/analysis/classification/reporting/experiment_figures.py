"""
Summary figures and tables of an experiment's classification runs: for one
feature version and model, the newest run of every (identification tool,
modality) -- single-modal -- and every (identification tool, fusion) --
multimodal -- is collected from its ``metadata.json``, and these are written:

    metrics.csv                              every metric of every run (tidy)
    metrics_<kind>_<validator>.pdf           heatmap per metric: tool x modality/fusion
    confusion_matrices_<kind>_<validator>.pdf
    misclassified_<kind>_<validator>.pdf     which samples are misclassified (rate)
    top_features_<kind>_<validator>.pdf      the 10 most important features per run

``kind`` is ``single`` or ``multi``. A missing run or validator leaves its
panel empty ("no run") rather than failing, so figures of a partial
experiment can still be made.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from phantom.config.classes import class_names
from phantom.config.loader import ConfigLoader

TIMESTAMP = re.compile(r"(\d{8}_\d{6})$")
METRICS = (
    "roc_auc",
    "pr_auc",
    "mcc",
    "bacc",
    "f1",
    "precision",
    "recall",
    "specificity",
    "npv",
    "gmean",
)
METRIC_NAMES = {
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC",
    "mcc": "MCC",
    "bacc": "Balanced accuracy",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall (sensitivity)",
    "specificity": "Specificity",
    "npv": "Negative predictive value",
    "gmean": "G-mean",
    "sensitivity": "Sensitivity",
}
VALIDATOR_NAMES = {"loocv": "leave-one-out CV", "rcv": "repeated stratified 5-fold CV"}
FUSION_NAMES = {"early": "Early fusion", "late": "Late fusion"}
TOOL_NAMES = {"genomad": "geNomad", "vibrant": "VIBRANT", "virsorter2": "VirSorter2"}
TOP_N = 10
CLASS_COLORS = {0: "#34495e", 1: "#27ae60", None: "#bdc3c7"}


@dataclass(frozen=True)
class Run:
    kind: str  # "single" | "multi"
    virus_identification_tool: str
    group: str  # modality (single) or fusion (multi)
    path: Path
    timestamp: str


@dataclass(frozen=True)
class Labels:
    """Display names: identification tools and modalities, from the configuration."""

    virus_identification_tools: dict[str, str]  # abbreviation -> display name, in config order
    modalities: dict[str, str]  # key -> label, in config order

    @classmethod
    def from_config(cls, config: dict) -> Labels:
        tools = ConfigLoader.virus_identification_tools(config)
        modalities = config.get("modalities", {})
        return cls(
            virus_identification_tools={
                abbreviation: TOOL_NAMES.get(name, name) for abbreviation, name in tools.items()
            },
            modalities={key: spec.get("label", key) for key, spec in modalities.items()},
        )

    def groups(self, kind: str) -> dict[str, str]:
        return self.modalities if kind == "single" else FUSION_NAMES


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _run_of(path: Path, kind: str, model: str) -> Run | None:
    metadata = _read_json(path / "metadata.json")
    if metadata is None or metadata.get("model_type") != model:
        return None
    match = TIMESTAMP.search(path.name)
    timestamp = match.group(1) if match else ""
    if kind == "single":
        virus_identification_tool, group = metadata.get("vtool"), metadata.get("modality")
    else:
        group = metadata.get("fusion")
        # Runs from before multimodal metadata recorded the tool: the feature
        # files are named <vtool>_<tag>_FEAT.csv.
        virus_identification_tool = (
            metadata.get("vtool") or Path(str(metadata.get("comp", ""))).name.split("_")[0]
        )
    if not virus_identification_tool or not group:
        return None
    return Run(kind, virus_identification_tool, group, path, timestamp)


def collect_runs(results_dir: Path, kind: str, model: str) -> dict[tuple[str, str], Run]:
    """The newest run of ``model`` per (vtool, group) under ``results_dir``."""
    newest: dict[tuple[str, str], Run] = {}
    if not results_dir.is_dir():
        return newest
    for path in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        run = _run_of(path, kind, model)
        if run is None:
            continue
        key = (run.virus_identification_tool, run.group)
        if key not in newest or run.timestamp >= newest[key].timestamp:
            newest[key] = run
    return newest


def metrics_table(runs: dict[tuple[str, str], Run], validators: list[str]) -> pd.DataFrame:
    rows = []
    for run in runs.values():
        for validator in validators:
            metrics = _read_json(run.path / validator / "metrics.json")
            if not metrics:
                continue
            for metric, value in metrics.items():
                score = value.get("score") if isinstance(value, dict) else value
                if isinstance(score, int | float):
                    rows.append(
                        {
                            "kind": run.kind,
                            "vtool": run.virus_identification_tool,
                            "group": run.group,
                            "validator": validator,
                            "metric": metric,
                            "score": float(score),
                            "run": run.path.name,
                        }
                    )
    columns = ["kind", "vtool", "group", "validator", "metric", "score", "run"]
    return pd.DataFrame(rows, columns=columns)


# plotting


def _tex(text: str) -> str:
    """Text safe for matplotlib, with or without LaTeX rendering."""
    import matplotlib as mpl

    if not mpl.rcParams.get("text.usetex"):
        return text
    for char in "\\&%$#_{}":
        text = text.replace(char, "\\" + char if char != "\\" else r"\textbackslash{}")
    return text


def _feature_label(name: str) -> str:
    """phavip ratios read as words (``lysis_ratio`` -> ``Lysis``); taxa unchanged."""
    if name.endswith("_ratio"):
        name = name.removesuffix("_ratio").replace("_", " ")
        return name[:1].upper() + name[1:]
    return name


def _empty(ax, text: str = "no run") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", color="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return path


def plot_metric_grid(table: pd.DataFrame, kind: str, validator: str, labels: Labels, path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    groups = labels.groups(kind)
    data = table[(table["kind"] == kind) & (table["validator"] == validator)]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8.5))
    for ax, metric in zip(axes.flat, METRICS, strict=True):
        subset = data[data["metric"] == metric]
        grid = (
            subset.pivot_table(index="vtool", columns="group", values="score")
            .reindex(index=list(labels.virus_identification_tools), columns=list(groups))
            .rename(index=labels.virus_identification_tools, columns=groups)
        )
        ax.set_title(_tex(METRIC_NAMES.get(metric, metric)), fontsize=13)
        if grid.isna().all().all():
            _empty(ax)
            continue
        sns.heatmap(
            grid,
            annot=True,
            fmt=".3f",
            cmap="Greens",
            cbar=False,
            linewidths=0.5,
            square=True,
            ax=ax,
            annot_kws={"size": 11},
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([_tex(t.get_text()) for t in ax.get_yticklabels()], rotation=0)
        ax.set_xticklabels([_tex(t.get_text()) for t in ax.get_xticklabels()], rotation=20)
    title = "Single-modal" if kind == "single" else "Multimodal"
    fig.suptitle(_tex(f"{title} classification, {VALIDATOR_NAMES[validator]}"), fontsize=18)
    fig.tight_layout()
    return _save(fig, path)


def _confusion(path: Path) -> np.ndarray | None:
    confusion = _read_json(path)
    if not confusion:
        return None
    return np.array(
        [
            [confusion.get("TN", 0), confusion.get("FP", 0)],
            [confusion.get("FN", 0), confusion.get("TP", 0)],
        ]
    )


def plot_confusion_grid(runs, kind: str, validator: str, labels: Labels, path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    virus_identification_tools, groups = (
        list(labels.virus_identification_tools),
        list(labels.groups(kind)),
    )
    fig, axes = plt.subplots(
        len(virus_identification_tools),
        len(groups),
        figsize=(3.4 * len(groups), 3.2 * len(virus_identification_tools)),
        squeeze=False,
    )
    for r, virus_identification_tool in enumerate(virus_identification_tools):
        for c, group in enumerate(groups):
            ax = axes[r, c]
            run = runs.get((virus_identification_tool, group))
            confusion = _confusion(run.path / validator / "confusion_matrix.json") if run else None
            if confusion is None:
                _empty(ax)
            else:
                sns.heatmap(
                    confusion,
                    annot=True,
                    fmt="d",
                    cmap="Greens",
                    cbar=False,
                    square=True,
                    linewidths=1,
                    linecolor="#979797",
                    xticklabels=["0", "1"],
                    yticklabels=["0", "1"],
                    annot_kws={"size": 16, "weight": "bold"},
                    ax=ax,
                )
                ax.set_xlabel("Predicted" if r == len(virus_identification_tools) - 1 else "")
                ax.set_ylabel("True")
                ax.tick_params(axis="y", labelrotation=0)
            if r == 0:
                ax.set_title(_tex(labels.groups(kind)[group]), fontsize=13)
            if c == 0:
                ax.annotate(
                    _tex(labels.virus_identification_tools[virus_identification_tool]),
                    xy=(-0.45, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=13,
                    fontweight="bold",
                )
    fig.suptitle(_tex(f"Confusion matrices, {VALIDATOR_NAMES[validator]}"), fontsize=16)
    fig.tight_layout()
    return _save(fig, path)


def _sample_number(sample: str) -> float:
    match = re.search(r"S(\d+)", sample)
    return float(match.group(1)) if match else float("inf")


def misclassification_rates(prediction_file: Path) -> pd.Series | None:
    """Per sample: the fraction of its predictions that were wrong (LOOCV: 0 or 1)."""
    if not prediction_file.is_file():
        return None
    df = pd.read_csv(prediction_file)
    if "sample_id" not in df.columns:
        return None
    sample = df["sample_id"].astype(str).str.split("|").str[-1]
    wrong = (df["y_true"] != df["y_pred"]).astype(float)
    return wrong.groupby(sample).mean()


def plot_misclassified(runs, kind: str, validator: str, labels: Labels, path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    virus_identification_tools, groups = (
        list(labels.virus_identification_tools),
        labels.groups(kind),
    )
    fig, axes = plt.subplots(
        len(virus_identification_tools),
        1,
        figsize=(19, 2.2 + 1.2 * len(groups) * len(virus_identification_tools)),
    )
    axes = np.atleast_1d(axes)
    for ax, virus_identification_tool in zip(axes, virus_identification_tools, strict=True):
        rows = {}
        for group, name in groups.items():
            run = runs.get((virus_identification_tool, group))
            rates = (
                misclassification_rates(run.path / validator / "predictions.csv") if run else None
            )
            if rates is not None:
                rows[name] = rates
        if not rows:
            _empty(ax)
            ax.set_ylabel(
                _tex(labels.virus_identification_tools[virus_identification_tool]),
                fontsize=13,
                fontweight="bold",
            )
            continue
        grid = pd.DataFrame(rows).T
        grid = grid[sorted(grid.columns, key=_sample_number)]
        sns.heatmap(
            grid,
            cmap="Greens",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="#f0f0f0",
            cbar_kws={"label": "misclassified (fraction of predictions)"},
            ax=ax,
        )
        ax.set_yticklabels([_tex(t.get_text()) for t in ax.get_yticklabels()], rotation=0)
        ax.tick_params(axis="x", labelsize=8, labelrotation=90)
        ax.set_xlabel("")
        ax.set_ylabel(
            _tex(labels.virus_identification_tools[virus_identification_tool]),
            fontsize=13,
            fontweight="bold",
        )
    axes[-1].set_xlabel("Samples")
    fig.suptitle(_tex(f"Misclassified samples, {VALIDATOR_NAMES[validator]}"), fontsize=16)
    fig.tight_layout()
    return _save(fig, path)


def _importance_files(run: Run, validator: str, labels: Labels) -> dict[str, Path]:
    """Single-modal: one file per run; multimodal: one per modality."""
    base = run.path / validator
    if run.kind == "single":
        return {run.group: base / "feature_importance.csv"}
    return {m: base / m / "feature_importance.csv" for m in labels.modalities}


def _top_features(file: Path | None) -> pd.DataFrame | None:
    """The TOP_N most important features, least important first (bar order);
    None without a file, empty when the model reported no importances."""
    if file is None or not file.is_file():
        return None
    df = pd.read_csv(file).dropna(subset=["importance"])
    return df.sort_values("importance", ascending=False).head(TOP_N).iloc[::-1]


def plot_top_features(runs, kind: str, validator: str, labels: Labels, path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    virus_identification_tools = list(labels.virus_identification_tools)
    # Columns: single-modal = modality; multimodal = fusion x modality.
    columns = (
        [(group, group) for group in labels.modalities]
        if kind == "single"
        else [(f, m) for f in FUSION_NAMES for m in labels.modalities]
    )
    fig, axes = plt.subplots(
        len(virus_identification_tools),
        len(columns),
        figsize=(4.6 * len(columns), 3.6 * len(virus_identification_tools)),
        squeeze=False,
    )
    for r, virus_identification_tool in enumerate(virus_identification_tools):
        for c, (group, modality) in enumerate(columns):
            ax = axes[r, c]
            run = runs.get((virus_identification_tool, group))
            file = _importance_files(run, validator, labels).get(modality) if run else None
            df = _top_features(file)
            if df is None:
                _empty(ax)
            elif df.empty:
                _empty(ax, "no importances")  # e.g. hgb under LOOCV
            else:
                higher = np.where(
                    df["mean_class_1"] > df["mean_class_0"],
                    1,
                    np.where(df["mean_class_0"] > df["mean_class_1"], 0, -1),
                )
                colors = [CLASS_COLORS.get(int(h), CLASS_COLORS[None]) for h in higher]
                names = [_tex(_feature_label(str(f))) for f in df["feature"]]
                ax.barh(names, df["importance"], color=colors, height=0.65)
                ax.spines[["top", "right"]].set_visible(False)
                ax.xaxis.grid(True, linestyle="--", alpha=0.5)
                ax.set_axisbelow(True)
                ax.tick_params(axis="y", labelsize=9)
                if r == len(virus_identification_tools) - 1:
                    ax.set_xlabel("importance")
            if r == 0:
                title = labels.modalities[modality]
                if kind == "multi":
                    title = f"{FUSION_NAMES[group]}: {title}"
                ax.set_title(_tex(title), fontsize=12)
            if c == 0:
                ax.set_ylabel(
                    _tex(labels.virus_identification_tools[virus_identification_tool]),
                    fontsize=13,
                    fontweight="bold",
                )
    classes = class_names()
    legend = [
        Patch(color=CLASS_COLORS[0], label=_tex(f"higher mean in {classes[0]} (0)")),
        Patch(color=CLASS_COLORS[1], label=_tex(f"higher mean in {classes[1]} (1)")),
        Patch(color=CLASS_COLORS[None], label="equal means"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(_tex(f"Top {TOP_N} features, {VALIDATOR_NAMES[validator]}"), y=1.04, fontsize=16)
    fig.tight_layout()
    return _save(fig, path)


def make_figures(
    results_dirs: dict[str, Path],
    model: str,
    validators: list[str],
    labels: Labels,
    out_dir: Path,
) -> list[Path]:
    """
    Every figure and the metrics table for the runs of ``model`` in
    ``results_dirs`` ({"single": <dir>, "multi": <dir>}). Returns the files
    written; raises if no run of the model exists at all.
    """
    from phantom.config.plotting import use_latex

    use_latex()
    runs = {kind: collect_runs(path, kind, model) for kind, path in results_dirs.items()}
    if not any(runs.values()):
        places = ", ".join(str(p) for p in results_dirs.values())
        raise FileNotFoundError(f"no {model} classification runs in {places}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat([metrics_table(r, validators) for r in runs.values() if r], ignore_index=True)
    written = [out_dir / "metrics.csv"]
    table.to_csv(written[0], index=False)
    for kind, kind_runs in runs.items():
        if not kind_runs:
            continue
        ran = set(table.loc[table["kind"] == kind, "validator"])
        for validator in (v for v in validators if v in ran):  # no empty figures
            suffix = f"{kind}_{validator}.pdf"
            written += [
                plot_metric_grid(table, kind, validator, labels, out_dir / f"metrics_{suffix}"),
                plot_confusion_grid(
                    kind_runs, kind, validator, labels, out_dir / f"confusion_matrices_{suffix}"
                ),
                plot_misclassified(
                    kind_runs, kind, validator, labels, out_dir / f"misclassified_{suffix}"
                ),
                plot_top_features(
                    kind_runs, kind, validator, labels, out_dir / f"top_features_{suffix}"
                ),
            ]
    return written
