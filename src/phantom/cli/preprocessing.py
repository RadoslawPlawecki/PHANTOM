"""
Contains CLI prompts used during the raw data preprocessing workflow.
"""

from pathlib import Path

import questionary

from phantom.config.loader import ConfigLoader

# Plot name (as MetadataVisualizer knows it) -> menu label.
PLOT_LABELS = {
    "checkv_quality": "CheckV genome quality per sample",
    "runtimes": "Tool runtime distribution",
    "size_vs_runtime": "Assembly size vs. runtime",
}


class PreprocessingPrompts:
    @staticmethod
    def ask_action() -> str:
        return questionary.select(
            "Select preprocessing action:",
            choices=[
                "1) Collect putative viral genomes",
                "2) Gather metadata",
                "3) Generate CheckV masks",
                "4) Visualize metadata",
                "Back",
            ],
        ).ask()

    @staticmethod
    def ask_plots(available: list) -> list:
        choices = [
            questionary.Choice(PLOT_LABELS[name], value=name, checked=True) for name in available
        ]
        return questionary.checkbox(
            "Select metadata plots to render (Space to select, Enter to confirm):",
            choices=choices,
        ).ask()

    @staticmethod
    def ask_plot_out_dir() -> Path | None:
        default_dir = ConfigLoader.resolve_data_path("data/plots/metadata")
        answer = questionary.path(
            "Output directory for plots:",
            default=str(default_dir),
            only_directories=True,
        ).ask()
        return Path(answer) if answer else None

    @staticmethod
    def ask_metadata_targets() -> list:
        return questionary.checkbox(
            "Select metadata to append/gather (Space to select, Enter to confirm):",
            choices=[
                "1) Renaming (format file paths)",
                "2) Labeling (assign labels to samples)",
                "3) Runtimes",
                "4) CheckV stats",
                "5) Input file sizes",
            ],
        ).ask()

    @staticmethod
    def ask_auto_format(invalid_count: int) -> bool:
        return questionary.confirm(
            f"Found {invalid_count} improperly named folders. Auto-format them?"
        ).ask()

    @staticmethod
    def ask_strategy(num_files: int) -> str:
        return questionary.select(
            f"Select labeling strategy for {num_files} files:",
            choices=[
                "1) Interactive Selection (manual)",
                "2) Sample ID Threshold (e.g., S <= n)",
                "3) Pattern Match (regex on original file name)",
                "4) External Manifest (CSV import)",
            ],
        ).ask()

    @staticmethod
    def ask_interactive(choices: list) -> list:
        return questionary.checkbox(
            "Select files to label as POSITIVE (Space to select, Enter to confirm):",
            choices=choices,
        ).ask()

    @staticmethod
    def ask_threshold() -> int:
        answer = questionary.text(
            "Enter threshold sample number (S_cutoff):",
            validate=lambda text: text.isdigit() or "Please enter a valid integer.",
        ).ask()
        return int(answer)

    @staticmethod
    def ask_pattern() -> str:
        return questionary.text("Enter positive match regex pattern:").ask()

    @staticmethod
    def ask_external_csv() -> str:
        return questionary.text(
            "Path to external CSV file (columns: id, label -- e.g. S12, 1):"
        ).ask()

    @staticmethod
    def ask_overwrite_labels(n_labeled: int, n_positive: int) -> bool:
        return questionary.confirm(
            f"Labels already exist ({n_labeled} samples, {n_positive} positives). Overwrite them?",
            default=False,
        ).ask()
