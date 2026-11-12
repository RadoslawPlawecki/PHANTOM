"""
Renders plots from the gathered sample metadata (the ``id``-keyed CSV managed by
``MetadataTable``): CheckV quality counts, tool runtimes and the input-size vs.
runtime relationship.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from phantom.config.plotting import use_latex

TOOLS = {"vib": "VIBRANT", "vs2": "VirSorter2", "gen": "geNomad"}

# Per tool: (all-passing, medium-quality, high-quality) bar shades, light -> dark.
# CheckV counts are cumulative (high-quality genomes are also medium-quality), so
# the bars are drawn back-to-front rather than side by side.
_QUALITY_PALETTE = {
    "VIBRANT": ("#BFE7DB", "#66C2A5", "#3B8C74"),
    "VirSorter2": ("#FDD1C2", "#FC8D62", "#C85A2E"),
    "geNomad": ("#CCD5EE", "#8DA0CB", "#5A6E9E"),
}

_TOOL_PALETTE = {name: shades[1] for name, shades in _QUALITY_PALETTE.items()}

# (count column, shade index into _QUALITY_PALETTE, legend label), light -> dark.
_QUALITY_TIERS = (
    ("Total", 0, "Total"),
    ("Medium", 1, "Medium-quality"),
    ("High", 2, "High-quality"),
)
# Neutral ramp used only to key the light->dark shading in the legend, so it
# reads as "darker = better tier" independently of any one tool's colour.
_SHADE_KEY = ("#C9C9C9", "#8A8A8A", "#3D3D3D")

_RUNTIME_COLS = {"vib_runtime": "VIBRANT", "vs2_runtime": "VirSorter2", "gen_runtime": "geNomad"}


class MetadataPlots:
    @staticmethod
    def available_plots(df: pd.DataFrame) -> list[str]:
        """Names of the plots the given metadata frame has the columns for."""
        plots = []
        if any(f"checkv_{tag}_total" in df.columns for tag in TOOLS):
            plots.append("checkv_quality")
        if any(col in df.columns for col in _RUNTIME_COLS):
            plots.append("runtimes")
        if "megahit_size_bytes" in df.columns and any(c in df.columns for c in _RUNTIME_COLS):
            plots.append("size_vs_runtime")
        return plots

    @staticmethod
    def render(df: pd.DataFrame, names: list[str], out_dir: Path) -> list[Path]:
        """Renders each named plot into ``out_dir``; returns the files written."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dispatch = {
            "checkv_quality": ("checkv_quality_bar.pdf", MetadataPlots.plot_checkv_quality),
            "runtimes": ("runtime_box.pdf", MetadataPlots.plot_runtimes),
            "size_vs_runtime": ("size_vs_runtime.pdf", MetadataPlots.plot_size_vs_runtime),
        }
        written = []
        for name in names:
            filename, plotter = dispatch[name]
            save_path = out_dir / filename
            plotter(df, save_path)
            written.append(save_path)
        return written

    @staticmethod
    def _checkv_long(df: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for tag, name in TOOLS.items():
            cols = [f"checkv_{tag}_{tier}" for tier in ("total", "medium", "high")]
            if not all(c in df.columns for c in cols):
                continue
            sub = df[["id", *cols]].copy()
            sub.columns = ["Sample", "Total", "Medium", "High"]
            sub["Tool"] = name
            frames.append(sub)
        if not frames:
            return pd.DataFrame(columns=["Sample", "Total", "Medium", "High", "Tool"])
        return pd.concat(frames, ignore_index=True).dropna(subset=["Total"])

    @staticmethod
    def plot_checkv_quality(df: pd.DataFrame, save_path: Path | None = None) -> None:
        """
        Grouped bars of CheckV genome counts per sample and tool, one row per half
        of the samples.
        """
        use_latex()
        long_df = MetadataPlots._checkv_long(df)
        if long_df.empty:
            return

        samples = list(dict.fromkeys(long_df["Sample"]))
        mid = (len(samples) + 1) // 2
        groups = [samples[:mid], samples[mid:]] if samples[mid:] else [samples]

        fig, axes = plt.subplots(len(groups), 1, figsize=(14, 5 * len(groups)), squeeze=False)
        for ax, group in zip(axes[:, 0], groups, strict=True):
            subset = long_df[long_df["Sample"].isin(group)]
            for tier, shade_idx, _ in _QUALITY_TIERS:
                palette = {name: shades[shade_idx] for name, shades in _QUALITY_PALETTE.items()}
                sns.barplot(
                    data=subset,
                    x="Sample",
                    y=tier,
                    hue="Tool",
                    order=group,
                    hue_order=list(TOOLS.values()),
                    palette=palette,
                    ax=ax,
                )
            ax.set_yscale("log")
            ax.set_ylim(bottom=1)
            ax.tick_params(axis="x", rotation=45, labelsize=12)
            ax.set_xlabel("Sample", fontsize=14, labelpad=10)
            ax.set_ylabel("Number of putative viral genomes", fontsize=14, labelpad=10)
            ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            if ax.get_legend():
                ax.get_legend().remove()

        tool_handles = [
            plt.Rectangle((0, 0), 1, 1, color=color, label=name)
            for name, color in _TOOL_PALETTE.items()
        ]
        shade_handles = [
            plt.Rectangle((0, 0), 1, 1, color=color, label=label)
            for color, (_, _, label) in zip(_SHADE_KEY, _QUALITY_TIERS, strict=True)
        ]
        tool_legend = axes[0, 0].legend(
            handles=tool_handles,
            title="Tool",
            loc="upper left",
            fontsize=10,
            frameon=True,
        )
        axes[0, 0].add_artist(tool_legend)
        axes[0, 0].legend(
            handles=shade_handles,
            title=r"Bar shade (light $\to$ dark)",
            loc="upper right",
            fontsize=10,
            frameon=True,
        )
        fig.suptitle(r"Genome Quality Control with CheckV", fontsize=16, y=0.96)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        MetadataPlots._finish(fig, save_path)

    @staticmethod
    def plot_runtimes(df: pd.DataFrame, save_path: Path | None = None) -> None:
        """Box plot of per-sample runtime for each viral identification tool."""
        use_latex()
        cols = [c for c in _RUNTIME_COLS if c in df.columns]
        if not cols:
            return
        melted = (
            df[cols]
            .melt(var_name="Tool", value_name="Runtime [min]")
            .dropna(subset=["Runtime [min]"])
        )
        melted["Tool"] = melted["Tool"].map(_RUNTIME_COLS)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=melted,
            x="Tool",
            y="Runtime [min]",
            hue="Tool",
            order=list(_RUNTIME_COLS.values()),
            palette=_TOOL_PALETTE,
            legend=False,
            width=0.5,
            linewidth=1.2,
            flierprops={"marker": "x"},
            ax=ax,
        )
        ax.set_title(r"Viral identification tool runtimes", fontsize=16, pad=10)
        ax.set_xlabel("Tool", fontsize=14, labelpad=10)
        ax.set_ylabel("Runtime [min]", fontsize=14, labelpad=10)
        fig.tight_layout()
        MetadataPlots._finish(fig, save_path)

    @staticmethod
    def plot_size_vs_runtime(df: pd.DataFrame, save_path: Path | None = None) -> None:
        """Scatter of tool runtime against assembly size, one point per run."""
        use_latex()
        cols = [c for c in _RUNTIME_COLS if c in df.columns]
        if "megahit_size_bytes" not in df.columns or not cols:
            return
        ordered = df.sort_values("megahit_size_bytes")
        frames = []
        for col in cols:
            sub = ordered[["id", "megahit_size_bytes", col]].copy()
            sub.columns = ["Sample", "Size", "Runtime"]
            sub["Tool"] = _RUNTIME_COLS[col]
            frames.append(sub)
        plot_df = pd.concat(frames, ignore_index=True).dropna(subset=["Runtime"])

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.scatterplot(
            data=plot_df,
            x="Sample",
            y="Runtime",
            hue="Tool",
            size="Size",
            hue_order=list(_RUNTIME_COLS.values()),
            sizes=(20, 500),
            alpha=0.6,
            palette=_TOOL_PALETTE,
            edgecolor="w",
            linewidth=0.5,
            ax=ax,
        )
        ax.set_title(r"Assembly size vs. tool runtime", fontsize=16)
        ax.set_xlabel(r"Samples (sorted ascending by assembly size)", fontsize=14, labelpad=10)
        ax.set_ylabel(r"Runtime [min]", fontsize=14, labelpad=10)
        ax.tick_params(axis="x", rotation=90, labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[: len(cols) + 1], labels[: len(cols) + 1], loc="upper left")
        fig.tight_layout()
        MetadataPlots._finish(fig, save_path)

    @staticmethod
    def _finish(fig, save_path: Path | None) -> None:
        if save_path is not None:
            fig.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
