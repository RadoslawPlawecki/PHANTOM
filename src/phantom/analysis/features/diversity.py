"""
Computes per-sample taxonomic (phagcn genus) and bacterial-host (cherry
ncbi_genus) diversity from a feature version's preprocessed data, split by
cases (label 1) vs. controls (label 0); ``[classes]`` names them.
"""

from pathlib import Path

import pandas as pd

from phantom.analysis.classification.data.labeling import Labeling
from phantom.analysis.features.diversity_plots import DiversityVisualizer
from phantom.analysis.features.phabox2.matrix import derive_patient_id
from phantom.config.classes import class_names, slug
from phantom.config.feature_versions import FeatureConfigManager

TOOL_FEATURE_COL = {"phagcn": "genus", "cherry": "ncbi_genus"}
TOOL_TITLES = {"phagcn": "Phage genus", "cherry": "Bacterial host genus"}
TOOL_FILE_STEMS = {"phagcn": "phagcn_genus", "cherry": "cherry_host"}
TOOL_UNIT_NAMES = {"phagcn": "genus", "cherry": "host"}
# Per-genome diversity variants: "ratio_of_means" = mean(units) / mean(genomes)
# per group (no whisker); "mean_of_ratios" = mean(units / genomes) over the
# group's samples (SD whisker).
RATIO_MODES = ("ratio_of_means", "mean_of_ratios")


class DiversityAnalyzer:
    def __init__(self, version: str, feature_version_manager: FeatureConfigManager | None = None):
        self.version = version
        self.feature_version_manager = feature_version_manager or FeatureConfigManager()
        self.preprocessed_dir = self.feature_version_manager.get_stage_dir(version, "preprocessed")

    def _load_virus_identification_tool_files(self, tool: str) -> dict[str, pd.DataFrame]:
        tool_dir = self.preprocessed_dir / tool
        loaded: dict[str, pd.DataFrame] = {}
        if not tool_dir.is_dir():
            return loaded
        for preprocessed_file in sorted(tool_dir.glob("*.csv")):
            virus_identification_tool = preprocessed_file.stem.split("_")[0]
            df = pd.read_csv(preprocessed_file, sep=";")
            df["id"] = derive_patient_id(df["Accession"])
            df["label"] = Labeling.default().derive_label(df["Accession"])
            loaded[virus_identification_tool] = df
        return loaded

    def analyze_tool(self, tool: str) -> dict[str, dict]:
        """
        Per vtool: {"per_sample": df[id, label, genome_count, unit_count,
        ratio], "group_stats": df indexed by label (0/1) with MultiIndex
        columns (metric, stat): ("unit_count", "mean"/"std"),
        ("genome_count", "mean"/"std"), ("mean_of_ratios", "mean"/"std")
        and ("ratio_of_means", "mean")}.

        ``unit_count`` is the number of *distinct* taxonomic units
        (occurrence-based, i.e. nunique over the feature column) present in
        a sample -- not a sum of per-unit abundance/occurrence counts.

        Two per-genome variants (see ``RATIO_MODES``):
        ``mean_of_ratios`` averages each sample's unit_count / genome_count
        over the group's samples; ``ratio_of_means`` is mean(unit_count) /
        mean(genome_count) over the group, which has no per-sample spread
        and therefore no "std".
        """
        feature_col = TOOL_FEATURE_COL[tool]
        results = {}
        for virus_identification_tool, df in self._load_virus_identification_tool_files(
            tool
        ).items():
            per_sample = df.groupby("id").agg(
                genome_count=("Accession", "nunique"),
                unit_count=(feature_col, "nunique"),
                label=("label", "first"),
            )
            per_sample["ratio"] = per_sample["unit_count"] / per_sample["genome_count"]
            group_stats = per_sample.groupby("label")[["unit_count", "genome_count", "ratio"]].agg(
                ["mean", "std"]
            )
            group_stats = group_stats.rename(columns={"ratio": "mean_of_ratios"}, level=0)
            group_stats[("ratio_of_means", "mean")] = (
                group_stats[("unit_count", "mean")] / group_stats[("genome_count", "mean")]
            )
            results[virus_identification_tool] = {
                "per_sample": per_sample.reset_index(),
                "group_stats": group_stats,
            }
        return results

    def case_only_units(self, tool: str) -> pd.DataFrame:
        """
        Columns: vtool, unit, n_samples, samples -- taxonomic units (phagcn
        genus / cherry host genus) whose sample-id set never intersects
        that vtool's control (label 0) sample-id set. ``samples`` lists the actual
        sample numbers (comma-separated) the unit was found in, alongside
        the plain count in ``n_samples``.
        """
        feature_col = TOOL_FEATURE_COL[tool]
        rows = []
        for virus_identification_tool, df in self._load_virus_identification_tool_files(
            tool
        ).items():
            control_ids = set(df.loc[df["label"] == 0, "id"])
            grouped = df.groupby(feature_col)["id"].agg(set)
            for unit, ids in grouped.items():
                if ids and not (ids & control_ids):
                    sample_numbers = sorted(int(i.split("|S")[1]) for i in ids)
                    rows.append(
                        {
                            "vtool": virus_identification_tool,
                            "unit": unit,
                            "n_samples": len(sample_numbers),
                            "samples": ",".join(str(n) for n in sample_numbers),
                        }
                    )
        columns = ["vtool", "unit", "n_samples", "samples"]
        if not rows:
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(rows, columns=columns)
            .sort_values(["vtool", "n_samples"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def run(self, out_dir: Path, ratio_mode: str = "ratio_of_means") -> None:
        if ratio_mode not in RATIO_MODES:
            raise ValueError(f"Unknown ratio mode {ratio_mode!r}, expected one of {RATIO_MODES}")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        any_tool_processed = False
        for tool in ("phagcn", "cherry"):
            results = self.analyze_tool(tool)
            if not results:
                print(
                    f"[WARNING] No preprocessed {tool} data found for version "
                    f"{self.version}, skipping."
                )
                continue
            any_tool_processed = True
            group_stats_by_virus_identification_tool = {
                virus_identification_tool: r["group_stats"]
                for virus_identification_tool, r in results.items()
            }
            unit_label = TOOL_TITLES[tool]
            file_stem = TOOL_FILE_STEMS[tool]
            unit_lower = unit_label.lower()
            if ratio_mode == "ratio_of_means":
                per_genome_ylabel = (
                    f"(Avg. distinct {unit_lower} / sample) / (Avg. genomes / sample)"
                )
            else:
                per_genome_ylabel = f"Avg. (distinct {unit_lower} / genomes) / sample"
            DiversityVisualizer.plot_group_bar_by_virus_identification_tool(
                group_stats_by_virus_identification_tool,
                value_col=ratio_mode,
                title=f"{unit_label} diversity per identified genome",
                ylabel=per_genome_ylabel,
                save_path=out_dir / f"{file_stem}_per_genome_{ratio_mode}.pdf",
            )
            DiversityVisualizer.plot_group_bar_by_virus_identification_tool(
                group_stats_by_virus_identification_tool,
                value_col="unit_count",
                title=f"{unit_label} diversity per sample",
                ylabel=f"Avg. distinct {unit_lower} / sample",
                save_path=out_dir / f"{file_stem}_per_sample.pdf",
            )
        if not any_tool_processed:
            print(
                f"[ERROR] No preprocessed phagcn/cherry data found under "
                f"{self.preprocessed_dir}. Run preprocessing first."
            )
            return

        case = slug(class_names()[1])
        for tool in ("phagcn", "cherry"):
            case_only = self.case_only_units(tool)
            case_only.to_csv(
                out_dir / f"{tool}_{case}_only_{TOOL_UNIT_NAMES[tool]}.csv", sep=";", index=False
            )

        from phantom.analysis.classification.reporting.result_files import ResultFileWriter

        ResultFileWriter(exp_dir=str(out_dir)).save_metadata(
            {
                "version": self.version,
                "tools": ["phagcn", "cherry"],
                "ratio_mode": ratio_mode,
                "labels": Labeling.default().provenance(),
            }
        )

        print(f"\n[SUCCESS] Diversity analysis complete. Results saved to {out_dir}")
