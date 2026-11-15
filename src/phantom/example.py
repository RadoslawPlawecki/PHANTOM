"""
A synthetic example workspace: ``phantom example DIR`` writes 12 samples
(6 cases, 6 controls) as the external tools would have left them -- the
assemblies under arbitrary file names in data/raw/case/ and data/raw/control/,
geNomad / VIBRANT / VirSorter2 viral contigs, CheckV quality summaries,
PHANOTATE ORFs and the PhaBOX2 cherry / phagcn / phavip predictions -- so the whole Python part of
the pipeline (``runs/example.toml``) runs in a few minutes without the
external tools or their databases. Cases carry more of some phage genera,
hosts and gene functions than controls, so the classifiers have a signal to
find. Nothing in it is real data.

The same generator drives the end-to-end test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phantom.config.loader import ConfigLoader

N_CASES = N_CONTROLS = 6
CONTIGS_PER_SAMPLE = 30
TOOLS = {"geN": "genomad", "VIB": "vibrant", "VS2": "virsorter2"}
# Viral genera: the first three are more frequent in cases.
GENERA = ["Eponavirus", "Kuravirus", "Lubbockvirus", "Oengusvirus", "Pankowvirus", "Wilnyevirus"]
LINEAGE = (
    "acellular root:Viruses;realm:Duplodnaviria;kingdom:Heunggongvirae;"
    "phylum:Uroviricota;class:Caudoviricetes;genus:{genus}"
)
# Hosts (NCBI lineage): the first two are more frequent in cases.
HOSTS = [
    ("Bacteroidota", "Bacteroidia", "Bacteroidales", "Bacteroidaceae", "Bacteroides", "fragilis"),
    (
        "Pseudomonadota",
        "Gammaproteobacteria",
        "Enterobacterales",
        "Enterobacteriaceae",
        "Escherichia",
        "coli",
    ),
    (
        "Bacillota",
        "Clostridia",
        "Eubacteriales",
        "Oscillospiraceae",
        "Faecalibacterium",
        "prausnitzii",
    ),
    (
        "Actinomycetota",
        "Actinomycetes",
        "Bifidobacteriales",
        "Bifidobacteriaceae",
        "Bifidobacterium",
        "longum",
    ),
    (
        "Verrucomicrobiota",
        "Verrucomicrobiia",
        "Verrucomicrobiales",
        "Akkermansiaceae",
        "Akkermansia",
        "muciniphila",
    ),
]
# Gene annotations by function: lysogeny and lysis are more frequent in cases.
ANNOTATIONS = {
    "lysogeny": ["integrase", "repressor protein"],
    "lysis": ["holin", "endolysin"],
    "structural": ["major capsid protein", "tail fiber protein", "portal protein"],
    "packaging": ["terminase large subunit"],
    "replication": ["DNA polymerase", "helicase"],
    "unknown": ["hypothetical protein"],
}
ENRICHED_GENERA = 3  # the first 3 genera
ENRICHED_HOSTS = 2  # the first 2 hosts
ENRICHED_FUNCTIONS = ("lysogeny", "lysis")


@dataclass(frozen=True)
class Contig:
    name: str
    length: int
    genus: str
    host: tuple[str, ...] | None
    completeness: float
    contamination: float
    provirus: bool
    orfs: tuple[tuple[str, str], ...]  # (ORF id, annotation)


def _weights(n: int, enriched: int, case: bool) -> np.ndarray:
    w = np.ones(n)
    w[:enriched] = 3.0 if case else 0.5
    return w / w.sum()


def _sequence(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list("ACGT"), size=length))


def _contigs(rng: np.random.Generator, case: bool) -> list[Contig]:
    functions = list(ANNOTATIONS)
    fw = np.array([3.0 if (f in ENRICHED_FUNCTIONS) == case else 1.0 for f in functions])
    contigs = []
    for i in range(1, CONTIGS_PER_SAMPLE + 1):
        name = f"k141_{i}"
        length = int(rng.integers(900, 2400))
        genus = GENERA[rng.choice(len(GENERA), p=_weights(len(GENERA), ENRICHED_GENERA, case))]
        host = (
            HOSTS[rng.choice(len(HOSTS), p=_weights(len(HOSTS), ENRICHED_HOSTS, case))]
            if rng.random() < 0.8
            else None
        )
        orfs = []
        start = 1
        for _ in range(int(rng.integers(2, 5))):
            end = min(length, start + int(rng.integers(150, 450)))
            function = functions[rng.choice(len(functions), p=fw / fw.sum())]
            annotation = ANNOTATIONS[function][rng.integers(len(ANNOTATIONS[function]))]
            orfs.append((f"{name}:{start}-{end}(+)", annotation))
            start = end + 1
            if start >= length - 100:
                break
        contigs.append(
            Contig(
                name=name,
                length=length,
                genus=genus,
                host=host,
                completeness=round(float(rng.uniform(5, 100)), 2),
                contamination=0.0 if rng.random() < 0.85 else round(float(rng.uniform(10, 60)), 2),
                provirus=bool(rng.random() < 0.15),
                orfs=tuple(orfs),
            )
        )
    return contigs


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fasta(records) -> str:
    return "".join(f">{name}\n{seq}\n" for name, seq in records)


def _tsv(header: list[str], rows) -> str:
    lines = ["\t".join(header)] + ["\t".join(str(v) for v in row) for row in rows]
    return "\n".join(lines) + "\n"


def _tool_id(abbreviation: str, contig: str) -> str:
    return f"{contig}||full" if abbreviation == "VS2" else contig  # VirSorter2 suffixes its ids


def _write_sample(
    root: Path, sample_id: str, contigs: list[Contig], rng: np.random.Generator
) -> None:
    data = root / "data"
    seqs = {c.name: _sequence(rng, c.length) for c in contigs}
    for abbreviation, tool in TOOLS.items():
        found = [c for c in contigs if rng.random() < 0.75]
        records = [(_tool_id(abbreviation, c.name), seqs[c.name]) for c in found]
        base = f"{sample_id}_{abbreviation}"
        viral = {
            "genomad": data / "genomad" / base / f"{sample_id}_summary" / f"{sample_id}_virus.fna",
            "vibrant": data
            / "vibrant"
            / base
            / f"VIBRANT_{sample_id}"
            / f"VIBRANT_phages_{sample_id}"
            / f"{sample_id}.phages_combined.fna",
            "virsorter2": data / "virsorter2" / base / "final-viral-combined.fa",
        }[tool]
        _write(viral, _fasta(records))
        _write(
            data / "checkv" / tool / f"{base}_ChV" / "quality_summary.tsv",
            _tsv(
                [
                    "contig_id",
                    "contig_length",
                    "provirus",
                    "proviral_length",
                    "gene_count",
                    "viral_genes",
                    "host_genes",
                    "checkv_quality",
                    "miuvig_quality",
                    "completeness",
                    "completeness_method",
                    "contamination",
                    "kmer_freq",
                    "warnings",
                ],
                [
                    (
                        _tool_id(abbreviation, c.name),
                        c.length,
                        "Yes" if c.provirus else "No",
                        "NA",
                        len(c.orfs),
                        len(c.orfs),
                        0,
                        "High-quality"
                        if c.completeness >= 90
                        else "Medium-quality"
                        if c.completeness >= 50
                        else "Low-quality",
                        "Genome-fragment",
                        c.completeness,
                        "AAI-based (high-confidence)",
                        c.contamination,
                        1.0,
                        "",
                    )
                    for c in found
                ],
            ),
        )
        _write(
            data / "phabox2" / "phagcn" / tool / f"{base}_PGN.tsv",
            _tsv(
                [
                    "Accession",
                    "Length",
                    "Lineage",
                    "PhaGCNScore",
                    "Genus",
                    "GenusCluster",
                    "Prokaryotic virus (Bacteriophages and Archaeal virus)",
                ],
                [
                    (
                        _tool_id(abbreviation, c.name),
                        c.length,
                        LINEAGE.format(genus=c.genus),
                        "1.00;1.00;1.00;1.00;1.00;0.95",
                        c.genus,
                        "known_genus",
                        "Y",
                    )
                    for c in found
                ],
            ),
        )
        cherry_rows = []
        for c in found:
            if c.host is None:
                cherry_rows.append(
                    (_tool_id(abbreviation, c.name), c.length, "-", "-", "", "-", "-")
                )
                continue
            phylum, cls, order, family, genus, species = c.host
            lineage = (
                f"d__Bacteria;p__{phylum};c__{cls};o__{order};f__{family};"
                f"g__{genus};s__{genus} {species}"
            )
            cherry_rows.append(
                (
                    _tool_id(abbreviation, c.name),
                    c.length,
                    f"species:{genus} {species}",
                    "1.0",
                    "CRISPR-based (DB)",
                    lineage,
                    "Not found",
                )
            )
        _write(
            data / "phabox2" / "cherry" / tool / f"{base}_CHR.tsv",
            _tsv(
                [
                    "Accession",
                    "Length",
                    "Host",
                    "CHERRYScore",
                    "Method",
                    "Host_NCBI_lineage",
                    "Host_GTDB_lineage",
                ],
                cherry_rows,
            ),
        )
        # ORF ids carry the tool's contig id (VirSorter2: "k141_1||full:1-300(+)").
        orfs = [
            (_tool_id(abbreviation, c.name) + orf[len(c.name) :], ann)
            for c in found
            for orf, ann in c.orfs
        ]
        _write(
            data / "phanotate" / "orfs" / tool / f"{base}_PHA_ORFs.fna",
            _fasta((orf, _sequence(rng, 90)) for orf, _ in orfs),
        )
        _write(
            data / "phabox2" / "phavip" / tool / f"{base}_PHA_ORFs_PHV.tsv",
            _tsv(
                [
                    "Genome",
                    "ORF",
                    "Start",
                    "End",
                    "Strand",
                    "GC",
                    "Annotation",
                    "pident",
                    "coverage",
                    "cloest_gene",
                ],
                [
                    (
                        orf,
                        f"{orf}_1",
                        1,
                        90,
                        1,
                        0.5,
                        ann,
                        f"{rng.uniform(40, 95):.2f}",
                        f"{rng.uniform(0.75, 1.0):.2f}",
                        "YP_000000.1",
                    )
                    for orf, ann in orfs
                ],
            ),
        )


def write_example(root: Path, seed: int = 7) -> Path:
    """Writes the example workspace into ``root`` (must be empty or absent);
    returns its config.toml."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} is not empty; choose a new folder")
    rng = np.random.default_rng(seed)
    config = ConfigLoader.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    config_path = root / "config.toml"
    _write(
        config_path,
        "# The example workspace: data and results live next to this file.\n"
        'workspace = "."\n\n' + config,
    )
    for n in range(1, N_CASES + N_CONTROLS + 1):
        case = n <= N_CASES
        sample_id = f"S{n}"
        contigs = _contigs(rng, case)
        _write_sample(root, sample_id, contigs, rng)
        # Assemblies under arbitrary names in a folder per class, as a user
        # would put them; preprocessing.samples numbers them in path order.
        name = f"{'case' if case else 'control'}/patient_{n:02d}/final.contigs.fa"
        records = [(c.name, _sequence(rng, 60)) for c in contigs]
        _write(root / "data" / "raw" / name, _fasta(records))
    return config_path
