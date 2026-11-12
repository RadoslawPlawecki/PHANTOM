"""
External-tool stages: each stage runs one target
rule of ``workflow/Snakefile`` through Snakemake, which skips outputs that are
up to date and runs independent samples in parallel. Everything
machine-specific (databases, conda environments, threads, tool checkouts)
comes from config.toml.

``tools.genomad`` / ``tools.vibrant`` / ``tools.virsorter2``  assemblies -> viral contigs
``tools.checkv``      genome quality per collected genome file
``tools.phanotate``   ORFs (gff -> bed -> fna -> faa) per genome file
``tools.phabox2``     cherry / phagcn on genomes, phavip on PHANOTATE ORFs
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from phantom.config.loader import ConfigLoader
from phantom.runner.params import param
from phantom.runner.stage import RunContext, Stage, register

IDENTIFIERS = ("genomad", "vibrant", "virsorter2")
PHABOX2_TASKS = ("cherry", "phagcn", "phavip")
SAMPLES_HELP = 'sample ids to run, e.g. ["S1", "S2"]; default: every sample found'
VTOOLS_HELP = "virus identification tools (keys of [tools]); default: all"
CORES_HELP = "Snakemake --cores (parallel jobs x threads); default: [workflow].cores"
DRY_RUN_HELP = "only print what Snakemake would run (nothing is executed or stamped)"
SETUP_TOOLS = (
    "genomad",
    "vibrant",
    "virsorter2",
    "checkv",
    "phanotate",
    "bedtools",
    "seqkit",
    "phabox2",
)


def _cfg(ctx: RunContext, section: str, key: str) -> Any:
    value = ctx.config.get(section, {}).get(key)
    if value is None:
        raise ValueError(f"config.toml is missing [{section}].{key}")
    return value


def _require(ctx: RunContext, keys: list[tuple[str, str]]) -> list[str]:
    errors = []
    for section, key in keys:
        if ctx.config.get(section, {}).get(key) is None:
            errors.append(f"config.toml is missing [{section}].{key}")
    return errors


def _vtool_names(ctx: RunContext, vtools: list[str] | None) -> list[str]:
    tools = ctx.config.get("tools", {})
    return [tools[v] for v in (vtools or list(tools))]


def _check_vtools(ctx: RunContext, vtools: list[str] | None) -> list[str]:
    return [
        f"vtool {v!r} not in [tools] (available: {', '.join(ctx.vtools)})"
        for v in vtools or []
        if v not in ctx.vtools
    ]


def snakemake_config(
    ctx: RunContext, vtools: list[str] | None = None, samples: list[str] | None = None
) -> dict:
    """The --configfile Snakefile reads: absolute paths, envs, threads."""
    sec = ctx.section_path
    phanotate = ctx.config.get("phanotate", {})
    tags = {m["tool"]: m["tag"] for m in ctx.modalities.values()}
    return {
        "paths": {
            "assemblies": str(sec("assemblies")),
            **{name: str(sec(name)) for name in IDENTIFIERS},
            "genomes": str(sec("genomes")),
            "checkv": str(sec("checkv")),
            "phabox2": str(sec("phabox2")),
            "phanotate_orfs": str(sec("phanotate")),
            "phanotate_genes": str(ctx.resolve(phanotate.get("genes", ""))),
            "phanotate_bed": str(ctx.resolve(phanotate.get("bed", ""))),
            "phanotate_proteins": str(ctx.resolve(phanotate.get("proteins", ""))),
        },
        "databases": {k: str(ctx.resolve(v)) for k, v in ctx.config.get("databases", {}).items()},
        "software": {k: str(ctx.resolve(v)) for k, v in ctx.config.get("software", {}).items()},
        "envs": dict(ctx.config.get("envs", {})),
        "threads": dict(ctx.config.get("threads", {})),
        "tools": dict(ctx.config.get("tools", {})),
        "tags": tags,
        "vtools": _vtool_names(ctx, vtools),
        "samples": samples,
        "setup_dir": str(ctx.resolve(_cfg(ctx, "workflow", "workdir")) / "setup"),
    }


def snakemake_command(
    ctx: RunContext,
    target: str | list[str],
    config_file: Path,
    cores: int | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    command = list(_cfg(ctx, "workflow", "snakemake"))
    return [
        *command,
        "--snakefile",
        str(ConfigLoader.PROJECT_ROOT / _cfg(ctx, "workflow", "snakefile")),
        "--directory",
        str(ctx.resolve(_cfg(ctx, "workflow", "workdir"))),
        "--configfile",
        str(config_file),
        "--cores",
        str(cores or _cfg(ctx, "workflow", "cores")),
        "--use-conda",
        "--printshellcmds",
        *(extra or []),
        *([target] if isinstance(target, str) else target),
    ]


def run_snakemake(
    ctx: RunContext,
    target: str | list[str],
    vtools: list[str] | None = None,
    samples: list[str] | None = None,
    cores: int | None = None,
    extra: list[str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    workdir = ctx.resolve(_cfg(ctx, "workflow", "workdir"))
    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=workdir, delete=False) as f:
        json.dump(snakemake_config(ctx, vtools, samples), f, indent=1)
        config_file = Path(f.name)
    try:
        extra = [*(extra or []), *(["--dry-run"] if dry_run else [])]
        command = snakemake_command(ctx, target, config_file, cores, extra)
        print("[INFO] " + " ".join(command), flush=True)
        result = subprocess.run(command, check=False)  # noqa: S603 -- argv from config.toml
        if result.returncode != 0:
            raise RuntimeError(f"snakemake {target} failed (exit {result.returncode})")
        return result
    finally:
        config_file.unlink(missing_ok=True)


# --- virus identification -------------------------------------------------------


@dataclass(frozen=True)
class IdentifyParams:
    samples: list[str] | None = param(None, help=SAMPLES_HELP)
    cores: int | None = param(None, help=CORES_HELP)
    dry_run: bool = param(False, help=DRY_RUN_HELP)


class _Identify(Stage):
    tool: str
    Params = IdentifyParams

    def validate(self, p: IdentifyParams, ctx: RunContext) -> list[str]:
        # [envs] / [software] entries are optional: without them the pinned
        # workflow/envs/<tool>.yaml environment and the bioconda package are used.
        keys = [(self.tool, "path"), ("assemblies", "path")]
        keys += [("threads", self.tool), ("workflow", "snakemake")]
        keys += [("databases", "genomad")] if self.tool == "genomad" else []
        return _require(ctx, keys)

    def check_inputs(self, p: IdentifyParams, ctx: RunContext) -> list[str]:
        assemblies = ctx.section_path("assemblies")
        if not assemblies.is_dir() or not any(assemblies.glob("S*.fa")):
            return [f"no assemblies (S<n>.fa) in {assemblies} -- run preprocessing.samples first"]
        return []

    def label(self, p: IdentifyParams) -> str:
        return ", ".join(p.samples) if p.samples else "all samples"

    def inputs(self, p: IdentifyParams, ctx: RunContext) -> list[Path]:
        return [ctx.section_path("assemblies")]

    def writes(self, p: IdentifyParams, ctx: RunContext) -> list[Path]:
        return [ctx.section_path(self.tool)]

    def run(self, p: IdentifyParams, ctx: RunContext) -> list[Path] | None:
        run_snakemake(ctx, f"all_{self.tool}", samples=p.samples, cores=p.cores, dry_run=p.dry_run)
        return None if p.dry_run else [ctx.section_path(self.tool)]


@register
class Genomad(_Identify):
    name = "tools.genomad"
    summary = "geNomad end-to-end on every assembly"
    tool = "genomad"


@register
class Vibrant(_Identify):
    name = "tools.vibrant"
    summary = "VIBRANT on every assembly"
    tool = "vibrant"


@register
class Virsorter2(_Identify):
    name = "tools.virsorter2"
    summary = "VirSorter2 on every assembly"
    tool = "virsorter2"


# --- tools on collected genomes ----------------------------------------------


@dataclass(frozen=True)
class GenomeToolParams:
    vtools: list[str] | None = param(None, help=VTOOLS_HELP)
    samples: list[str] | None = param(None, help=SAMPLES_HELP)
    cores: int | None = param(None, help=CORES_HELP)
    dry_run: bool = param(False, help=DRY_RUN_HELP)


def _genome_dirs(ctx: RunContext, vtools: list[str] | None) -> list[Path]:
    return [ctx.section_path("genomes") / name for name in _vtool_names(ctx, vtools)]


def _check_genomes(ctx: RunContext, vtools: list[str] | None) -> list[str]:
    return [
        f"no collected genomes (*.fna) in {d} -- run preprocessing.collect first"
        for d in _genome_dirs(ctx, vtools)
        if not d.is_dir() or not any(d.glob("*.fna"))
    ]


@register
class Checkv(Stage):
    name = "tools.checkv"
    summary = "CheckV end-to-end on every collected genome file"
    Params = GenomeToolParams

    def validate(self, p: GenomeToolParams, ctx: RunContext) -> list[str]:
        keys = [("checkv", "path"), ("databases", "checkv")]
        return _check_vtools(ctx, p.vtools) + _require(ctx, [*keys, ("threads", "checkv")])

    def check_inputs(self, p: GenomeToolParams, ctx: RunContext) -> list[str]:
        return _check_genomes(ctx, p.vtools)

    def label(self, p: GenomeToolParams) -> str:
        return ", ".join(p.vtools) if p.vtools else "all vtools"

    def inputs(self, p: GenomeToolParams, ctx: RunContext) -> list[Path]:
        return _genome_dirs(ctx, p.vtools)

    def writes(self, p: GenomeToolParams, ctx: RunContext) -> list[Path]:
        return [ctx.section_path("checkv") / n for n in _vtool_names(ctx, p.vtools)]

    def run(self, p: GenomeToolParams, ctx: RunContext) -> list[Path] | None:
        run_snakemake(ctx, "all_checkv", p.vtools, p.samples, p.cores, dry_run=p.dry_run)
        return None if p.dry_run else [d for d in self.writes(p, ctx) if d.is_dir()]


@register
class Phanotate(Stage):
    name = "tools.phanotate"
    summary = "PHANOTATE ORFs per genome file: gff -> bed -> ORF fasta -> proteins"
    Params = GenomeToolParams

    def _dirs(self, ctx: RunContext) -> list[Path]:
        ph = ctx.config.get("phanotate", {})
        return [ctx.resolve(ph[k]) for k in ("genes", "bed", "proteins")] + [
            ctx.section_path("phanotate")
        ]

    def validate(self, p: GenomeToolParams, ctx: RunContext) -> list[str]:
        keys = [("phanotate", k) for k in ("path", "genes", "bed", "proteins")]
        return _check_vtools(ctx, p.vtools) + _require(ctx, keys)

    def check_inputs(self, p: GenomeToolParams, ctx: RunContext) -> list[str]:
        return _check_genomes(ctx, p.vtools)

    def label(self, p: GenomeToolParams) -> str:
        return ", ".join(p.vtools) if p.vtools else "all vtools"

    def inputs(self, p: GenomeToolParams, ctx: RunContext) -> list[Path]:
        return _genome_dirs(ctx, p.vtools)

    def writes(self, p: GenomeToolParams, ctx: RunContext) -> list[Path]:
        return [d / n for d in self._dirs(ctx) for n in _vtool_names(ctx, p.vtools)]

    def run(self, p: GenomeToolParams, ctx: RunContext) -> list[Path] | None:
        run_snakemake(ctx, "all_phanotate", p.vtools, p.samples, p.cores, dry_run=p.dry_run)
        return None if p.dry_run else [d for d in self.writes(p, ctx) if d.is_dir()]


@dataclass(frozen=True)
class Phabox2Params:
    tasks: list[Literal["cherry", "phagcn", "phavip"]] = param(
        default_factory=lambda: list(PHABOX2_TASKS),
        help="cherry/phagcn run on genomes, phavip on PHANOTATE ORFs",
    )
    vtools: list[str] | None = param(None, help=VTOOLS_HELP)
    samples: list[str] | None = param(None, help=SAMPLES_HELP)
    cores: int | None = param(None, help=CORES_HELP)
    dry_run: bool = param(False, help=DRY_RUN_HELP)


@register
class Phabox2(Stage):
    name = "tools.phabox2"
    summary = "PhaBOX2 host (cherry), taxonomy (phagcn) and gene function (phavip)"
    Params = Phabox2Params

    def validate(self, p: Phabox2Params, ctx: RunContext) -> list[str]:
        errors = _check_vtools(ctx, p.vtools)
        if not p.tasks:
            errors.append("tasks: give at least one of cherry, phagcn, phavip")
        tags = {m["tool"] for m in ctx.modalities.values()}
        errors += [
            f"no [modalities] entry with tool = {t!r} (its filename tag)"
            for t in p.tasks
            if t not in tags
        ]
        keys = [
            ("phabox2", "path"),
            ("databases", "phabox2"),
            ("threads", "phabox2"),
        ]
        return errors + _require(ctx, keys)

    def check_inputs(self, p: Phabox2Params, ctx: RunContext) -> list[str]:
        errors = []
        if {"cherry", "phagcn"} & set(p.tasks):
            errors += _check_genomes(ctx, p.vtools)
        if "phavip" in p.tasks:
            orfs = ctx.section_path("phanotate")
            errors += [
                f"no PHANOTATE ORFs in {orfs / n} -- run tools.phanotate first"
                for n in _vtool_names(ctx, p.vtools)
                if not (orfs / n).is_dir() or not any((orfs / n).glob("*.fna"))
            ]
        return errors

    def label(self, p: Phabox2Params) -> str:
        return " + ".join(p.tasks)

    def inputs(self, p: Phabox2Params, ctx: RunContext) -> list[Path]:
        dirs = []
        if {"cherry", "phagcn"} & set(p.tasks):
            dirs += _genome_dirs(ctx, p.vtools)
        if "phavip" in p.tasks:
            dirs += [ctx.section_path("phanotate") / n for n in _vtool_names(ctx, p.vtools)]
        return dirs

    def writes(self, p: Phabox2Params, ctx: RunContext) -> list[Path]:
        root = ctx.section_path("phabox2")
        return [root / t / n for t in p.tasks for n in _vtool_names(ctx, p.vtools)]

    def run(self, p: Phabox2Params, ctx: RunContext) -> list[Path] | None:
        targets = [f"all_phabox2_{task}" for task in p.tasks]
        run_snakemake(ctx, targets, p.vtools, p.samples, p.cores, dry_run=p.dry_run)
        return None if p.dry_run else [d for d in self.writes(p, ctx) if d.is_dir()]


# --- installation -------------------------------------------------------------


@dataclass(frozen=True)
class SetupParams:
    tools: list[
        Literal[
            "genomad",
            "vibrant",
            "virsorter2",
            "checkv",
            "phanotate",
            "bedtools",
            "seqkit",
            "phabox2",
        ]
    ] = param(default_factory=lambda: list(SETUP_TOOLS), help="tools to install / check")
    dry_run: bool = param(False, help=DRY_RUN_HELP)


@register
class Setup(Stage):
    """
    For each tool: create its conda environment from workflow/envs/<tool>.yaml
    if [envs] names none, download its database if the [databases] path is
    missing, and check that it starts. Safe to rerun: existing environments
    and databases are reused, never replaced.
    """

    name = "tools.setup"
    summary = "Install external tools (conda envs + databases) and check they run"
    Params = SetupParams

    def validate(self, p: SetupParams, ctx: RunContext) -> list[str]:
        return _require(
            ctx, [("workflow", "snakemake"), ("workflow", "snakefile"), ("workflow", "workdir")]
        )

    def label(self, p: SetupParams) -> str:
        return ", ".join(p.tools)

    def inputs(self, p: SetupParams, ctx: RunContext) -> list[Path]:
        return []

    def writes(self, p: SetupParams, ctx: RunContext) -> list[Path]:
        return [ctx.resolve(_cfg(ctx, "workflow", "workdir")) / "setup"]

    def run(self, p: SetupParams, ctx: RunContext) -> list[Path] | None:
        setup_dir = ctx.resolve(_cfg(ctx, "workflow", "workdir")) / "setup"
        # The markers are cheap to recreate; always recheck that tools start.
        for tool in p.tools:
            (setup_dir / f"{tool}.ok").unlink(missing_ok=True)
        run_snakemake(ctx, [f"setup_{t}" for t in p.tools], dry_run=p.dry_run)
        return None if p.dry_run else [setup_dir]
