"""
Groups PHANOTATE-called ORFs by their predicted functional category and
writes, per category, a FASTA of the member ORFs plus a summary table
annotating each ORF with its contig's predicted genus and host.

Functional categories come from phavip, genus from phagcn and host from
cherry -- all read from the chosen feature version's ``preprocessed/``
stage, so the summary always reflects the same filtering the feature
matrices were built from. ORFs absent from the preprocessed phavip file
(dropped by masking/preprocessing) are kept and labelled ``uncategorized``.
"""

import csv
import re
from pathlib import Path

import pandas as pd

from phantom.config.feature_versions import FeatureConfigManager
from phantom.config.loader import ConfigLoader

ORF_TAG = "FUNC"
UNCATEGORIZED = "uncategorized"
UNKNOWN = "Unknown"

# phavip -> functional category, phagcn -> genus, cherry -> host taxonomy.
CATEGORY_TOOL = "phavip"
GENUS_TOOL = "phagcn"
HOST_TOOL = "cherry"


def normalize_accession(accession: str) -> str:
    """Collapses the legacy ``k_<n>`` contig spelling to ``k<n>``, matching
    the accession form produced by ``analysis/features/tools/utils.format_accession``."""
    return re.sub(r"k_(\d+)", r"k\1", accession.strip())


def iter_fasta(path: Path):
    """Yields ``(header, sequence)`` pairs from a FASTA file (header without '>')."""
    header: str | None = None
    seq: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header, seq = line[1:], []
            else:
                seq.append(line)
    if header is not None:
        yield header, "".join(seq)


class FunctionSummarizer:
    def __init__(self, version: str, config: dict, feature_version_manager: FeatureConfigManager):
        self.version = version
        self.config = config
        self.preprocessed_dir = feature_version_manager.get_stage_dir(version, "preprocessed")
        self.orf_root = ConfigLoader.config_path(config, "phanotate")
        self.orf_glob = config.get("phanotate", {}).get("orf_fna", "*_PHA_ORFs.fna")
        self.out_root = ConfigLoader.config_path(config, "functions") / version

        # config's [tools] maps abbreviation -> tool name (e.g. geN -> genomad);
        # ORF subdirectories are named after the tool, so invert it.
        self.abbr_by_tool = {
            name: abbreviation
            for abbreviation, name in ConfigLoader.virus_identification_tools(self.config).items()
        }

    def run(self) -> None:
        print(f"\n[INFO] Summarizing ORF functions for version {self.version}...")
        if not self.orf_root.exists():
            print(f"[ERROR] ORF directory does not exist: {self.orf_root}")
            return
        if not self.preprocessed_dir.exists():
            print(
                f"[ERROR] Preprocessed feature directory does not exist: "
                f"{self.preprocessed_dir}. Run feature preprocessing first."
            )
            return

        processed_any = False
        for tool_dir in sorted(p for p in self.orf_root.iterdir() if p.is_dir()):
            abbreviation = self.abbr_by_tool.get(tool_dir.name)
            if abbreviation is None:
                print(
                    f"[WARNING] Skipping '{tool_dir.name}': not registered in the "
                    f"[tools] section of config.toml."
                )
                continue
            if self._process_tool(tool_dir, abbreviation):
                processed_any = True

        if not processed_any:
            print(f"[ERROR] No ORF files matched under {self.orf_root}.")

    def _process_tool(self, tool_dir: Path, abbreviation: str) -> bool:
        orf_files = sorted(tool_dir.glob(self.orf_glob))
        if not orf_files:
            print(f"[INFO] No ORF files ({self.orf_glob}) in {tool_dir}.")
            return False

        annotations = self._load_lookup(CATEGORY_TOOL, abbreviation, "Category", contig_only=False)
        if annotations is None:
            print(
                f"[WARNING] Skipping '{tool_dir.name}': no preprocessed "
                f"{CATEGORY_TOOL} file for '{abbreviation}' under {self.preprocessed_dir}."
            )
            return False
        genus = self._load_lookup(GENUS_TOOL, abbreviation, "genus", contig_only=True) or {}
        host = self._load_host_lookup(abbreviation)

        for orf_file in orf_files:
            self._process_file(orf_file, tool_dir.name, abbreviation, annotations, genus, host)
        return True

    def _find_preprocessed(self, tool: str, abbreviation: str) -> Path | None:
        tool_dir = self.preprocessed_dir / tool
        if not tool_dir.is_dir():
            return None
        matches = sorted(tool_dir.glob(f"{abbreviation}_*.csv"))
        return matches[0] if matches else None

    def _load_lookup(
        self, tool: str, abbreviation: str, value_col: str, contig_only: bool
    ) -> dict[str, str] | None:
        path = self._find_preprocessed(tool, abbreviation)
        if path is None:
            return None
        df = pd.read_csv(path, sep=";")
        if "Accession" not in df.columns or value_col not in df.columns:
            print(f"[WARNING] {path.name} is missing 'Accession'/'{value_col}'.")
            return {}
        keys = df["Accession"].map(self._accession_key(contig_only))
        return dict(zip(keys, df[value_col].fillna(UNKNOWN), strict=True))

    def _load_host_lookup(self, abbreviation: str) -> dict[str, str]:
        path = self._find_preprocessed(HOST_TOOL, abbreviation)
        if path is None:
            return {}
        df = pd.read_csv(path, sep=";")
        if "Accession" not in df.columns:
            print(f"[WARNING] {path.name} is missing 'Accession'.")
            return {}
        genus = df.get("ncbi_genus", pd.Series("", index=df.index)).fillna("")
        species = df.get("ncbi_species", pd.Series("", index=df.index)).fillna("")
        host_info = (genus.astype(str) + " " + species.astype(str)).str.strip()
        host_info = host_info.replace("", UNKNOWN)
        keys = df["Accession"].map(self._accession_key(contig_only=True))
        return dict(zip(keys, host_info, strict=True))

    @staticmethod
    def _accession_key(contig_only: bool):
        def key(accession: str) -> str:
            accession = normalize_accession(str(accession))
            return accession.split(":", 1)[0] if contig_only else accession

        return key

    def _process_file(
        self,
        orf_file: Path,
        tool: str,
        abbreviation: str,
        annotations: dict[str, str],
        genus: dict[str, str],
        host: dict[str, str],
    ) -> None:
        sample_match = re.match(r"(S\d+)_", orf_file.name)
        if not sample_match:
            print(f"[WARNING] Cannot parse a sample id from {orf_file.name}; skipping.")
            return
        sample_id = sample_match.group(1)
        print(f"\n[INFO] Sorting {tool}/{orf_file.name}...")

        fasta_by_category: dict[str, list[str]] = {}
        summary_by_category: dict[str, list[list[str]]] = {}
        for header, sequence in iter_fasta(orf_file):
            accession = normalize_accession(f"{abbreviation}|{sample_id}_{header}")
            contig_id = accession.split(":", 1)[0]
            category = annotations.get(accession, UNCATEGORIZED)
            fasta_by_category.setdefault(category, []).append(f">{accession}\n{sequence}\n")
            summary_by_category.setdefault(category, []).append(
                [accession, genus.get(contig_id, UNKNOWN), host.get(contig_id, UNKNOWN)]
            )

        stem = f"{orf_file.stem}_{ORF_TAG}"
        total = 0
        for category, records in fasta_by_category.items():
            cat_dir = self.out_root / tool / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            (cat_dir / f"{stem}.fna").write_text("".join(records), encoding="utf-8")
            with open(cat_dir / f"{stem}.tsv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["Accession", "Genus", "Host"])
                writer.writerows(summary_by_category[category])
            total += len(records)
        print(
            f"       {total} ORFs -> {len(fasta_by_category)} categories "
            f"under {self.out_root / tool}"
        )
