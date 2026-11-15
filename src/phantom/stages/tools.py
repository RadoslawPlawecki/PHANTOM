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

import dataclasses
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from phantom.config.loader import ConfigLoader
from phantom.runner.parameters import param
from phantom.runner.stage import RunContext, Stage, register
from phantom.stages._common import resolve_virus_identification_tool

IDENTIFIERS = ("genomad", "vibrant", "virsorter2")
PHABOX2_TASKS = ("cherry", "phagcn", "phavip")
SAMPLES_HELP = 'sample ids to run, e.g. ["S1", "S2"]; default: every sample found'
VIRUS_IDENTIFICATION_TOOLS_HELP = (
    "virus identification tools, by abbreviation or full name "
    '(e.g. ["geN"] or ["genomad"]); default: all'
)
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


def _config_value(context: RunContext, section: str, key: str) -> Any:
    value = context.config.get(section, {}).get(key)
    if value is None:
        raise ValueError(f"config.toml is missing [{section}].{key}")
    return value


def _require(context: RunContext, keys: list[tuple[str, str]]) -> list[str]:
    errors = []
    for section, key in keys:
        if context.config.get(section, {}).get(key) is None:
            errors.append(f"config.toml is missing [{section}].{key}")
    return errors


def _virus_identification_tool_names(
    context: RunContext, virus_identification_tools: list[str] | None
) -> list[str]:
    tools = ConfigLoader.virus_identification_tools(context.config)
    return [tools[v] for v in (virus_identification_tools or list(tools))]


def _resolve_virus_identification_tools(params, context: RunContext):
    """Full tool names in ``vtools`` -> abbreviations (one job, one stamp either way)."""
    if params.vtools is None:
        return params
    return dataclasses.replace(
        params, vtools=[resolve_virus_identification_tool(context, v) for v in params.vtools]
    )


def _check_virus_identification_tools(
    context: RunContext, virus_identification_tools: list[str] | None
) -> list[str]:
    return [
        f"vtool {v!r} is not a virus identification tool (available: {', '.join(context.virus_identification_tools)})"
        for v in virus_identification_tools or []
        if v not in context.virus_identification_tools
    ]


def snakemake_config(
    context: RunContext,
    virus_identification_tools: list[str] | None = None,
    samples: list[str] | None = None,
) -> dict:
    """The --configfile Snakefile reads: absolute paths, envs, threads."""
    sec = context.section_path
    phanotate = context.config.get("phanotate", {})
    tags = {modality["tool"]: modality["tag"] for modality in context.modalities.values()}
    return {
        "paths": {
            "assemblies": str(sec("assemblies")),
            **{name: str(sec(name)) for name in IDENTIFIERS},
            "genomes": str(sec("genomes")),
            "checkv": str(sec("checkv")),
            "phabox2": str(sec("phabox2")),
            "phanotate_orfs": str(sec("phanotate")),
            "phanotate_genes": str(context.resolve(phanotate.get("genes", ""))),
            "phanotate_bed": str(context.resolve(phanotate.get("bed", ""))),
            "phanotate_proteins": str(context.resolve(phanotate.get("proteins", ""))),
        },
        "databases": {
            k: str(context.resolve(v)) for k, v in context.config.get("databases", {}).items()
        },
        "software": {
            k: str(context.resolve(v)) for k, v in context.config.get("software", {}).items()
        },
        "envs": dict(context.config.get("envs", {})),
        "threads": dict(context.config.get("threads", {})),
        "virus_identification_tools": ConfigLoader.virus_identification_tools(context.config),
        "filename_tags": tags,
        "selected_virus_identification_tools": _virus_identification_tool_names(
            context, virus_identification_tools
        ),
        "samples": samples,
        "setup_dir": str(context.resolve(_config_value(context, "workflow", "workdir")) / "setup"),
    }


def snakemake_command(
    context: RunContext,
    target: str | list[str],
    config_file: Path,
    cores: int | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    command = list(_config_value(context, "workflow", "snakemake"))
    return [
        *command,
        "--snakefile",
        str(ConfigLoader.PROJECT_ROOT / _config_value(context, "workflow", "snakefile")),
        "--directory",
        str(context.resolve(_config_value(context, "workflow", "workdir"))),
        "--configfile",
        str(config_file),
        "--cores",
        str(cores or _config_value(context, "workflow", "cores")),
        "--use-conda",
        "--printshellcmds",
        *(extra or []),
        *([target] if isinstance(target, str) else target),
    ]


def run_snakemake(
    context: RunContext,
    target: str | list[str],
    virus_identification_tools: list[str] | None = None,
    samples: list[str] | None = None,
    cores: int | None = None,
    extra: list[str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    workdir = context.resolve(_config_value(context, "workflow", "workdir"))
    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=workdir, delete=False) as f:
        json.dump(snakemake_config(context, virus_identification_tools, samples), f, indent=1)
        config_file = Path(f.name)
    try:
        extra = [*(extra or []), *(["--dry-run"] if dry_run else [])]
        command = snakemake_command(context, target, config_file, cores, extra)
        print("[INFO] " + " ".join(command), flush=True)
        result = subprocess.run(command, check=False)  # noqa: S603 -- argv from config.toml
        if result.returncode != 0:
            raise RuntimeError(f"snakemake {target} failed (exit {result.returncode})")
        return result
    finally:
        config_file.unlink(missing_ok=True)


# virus identification


@dataclass(frozen=True)
class IdentifyParams:
    samples: list[str] | None = param(None, help=SAMPLES_HELP)
    cores: int | None = param(None, help=CORES_HELP)
    dry_run: bool = param(False, help=DRY_RUN_HELP)


class _Identify(Stage):
    tool: str
    Params = IdentifyParams

    def validate(self, params: IdentifyParams, context: RunContext) -> list[str]:
        # [envs] / [software] entries are optional: without them the pinned
        # workflow/envs/<tool>.yaml environment and the bioconda package are used.
        keys = [(self.tool, "path"), ("assemblies", "path")]
        keys += [("threads", self.tool), ("workflow", "snakemake")]
        keys += [("databases", "genomad")] if self.tool == "genomad" else []
        return _require(context, keys)

    def check_inputs(self, params: IdentifyParams, context: RunContext) -> list[str]:
        assemblies = context.section_path("assemblies")
        if not assemblies.is_dir() or not any(assemblies.glob("S*.fa")):
            return [f"no assemblies (S<n>.fa) in {assemblies} -- run preprocessing.samples first"]
        return []

    def label(self, params: IdentifyParams) -> str:
        return ", ".join(params.samples) if params.samples else "all samples"

    def inputs(self, params: IdentifyParams, context: RunContext) -> list[Path]:
        return [context.section_path("assemblies")]

    def writes(self, params: IdentifyParams, context: RunContext) -> list[Path]:
        return [context.section_path(self.tool)]

    def run(self, params: IdentifyParams, context: RunContext) -> list[Path] | None:
        run_snakemake(
            context,
            f"all_{self.tool}",
            samples=params.samples,
            cores=params.cores,
            dry_run=params.dry_run,
        )
        return None if params.dry_run else [context.section_path(self.tool)]


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


# tools on collected genomes


@dataclass(frozen=True)
class GenomeToolParams:
    vtools: list[str] | None = param(None, help=VIRUS_IDENTIFICATION_TOOLS_HELP)
    samples: list[str] | None = param(None, help=SAMPLES_HELP)
    cores: int | None = param(None, help=CORES_HELP)
    dry_run: bool = param(False, help=DRY_RUN_HELP)


def _genome_dirs(context: RunContext, virus_identification_tools: list[str] | None) -> list[Path]:
    return [
        context.section_path("genomes") / name
        for name in _virus_identification_tool_names(context, virus_identification_tools)
    ]


def _check_genomes(context: RunContext, virus_identification_tools: list[str] | None) -> list[str]:
    return [
        f"no collected genomes (*.fna) in {d} -- run preprocessing.collect first"
        for d in _genome_dirs(context, virus_identification_tools)
        if not d.is_dir() or not any(d.glob("*.fna"))
    ]


@register
class Checkv(Stage):
    name = "tools.checkv"
    summary = "CheckV end-to-end on every collected genome file"
    Params = GenomeToolParams

    def resolve(self, params, context: RunContext):
        return _resolve_virus_identification_tools(params, context)

    def validate(self, params: GenomeToolParams, context: RunContext) -> list[str]:
        keys = [("checkv", "path"), ("databases", "checkv")]
        return _check_virus_identification_tools(context, params.vtools) + _require(
            context, [*keys, ("threads", "checkv")]
        )

    def check_inputs(self, params: GenomeToolParams, context: RunContext) -> list[str]:
        return _check_genomes(context, params.vtools)

    def label(self, params: GenomeToolParams) -> str:
        return ", ".join(params.vtools) if params.vtools else "all vtools"

    def inputs(self, params: GenomeToolParams, context: RunContext) -> list[Path]:
        return _genome_dirs(context, params.vtools)

    def writes(self, params: GenomeToolParams, context: RunContext) -> list[Path]:
        return [
            context.section_path("checkv") / n
            for n in _virus_identification_tool_names(context, params.vtools)
        ]

    def run(self, params: GenomeToolParams, context: RunContext) -> list[Path] | None:
        run_snakemake(
            context,
            "all_checkv",
            params.vtools,
            params.samples,
            params.cores,
            dry_run=params.dry_run,
        )
        return None if params.dry_run else [d for d in self.writes(params, context) if d.is_dir()]


@register
class Phanotate(Stage):
    name = "tools.phanotate"
    summary = "PHANOTATE ORFs per genome file: gff -> bed -> ORF fasta -> proteins"
    Params = GenomeToolParams

    def resolve(self, params, context: RunContext):
        return _resolve_virus_identification_tools(params, context)

    def _dirs(self, context: RunContext) -> list[Path]:
        ph = context.config.get("phanotate", {})
        return [context.resolve(ph[k]) for k in ("genes", "bed", "proteins")] + [
            context.section_path("phanotate")
        ]

    def validate(self, params: GenomeToolParams, context: RunContext) -> list[str]:
        keys = [("phanotate", k) for k in ("path", "genes", "bed", "proteins")]
        return _check_virus_identification_tools(context, params.vtools) + _require(context, keys)

    def check_inputs(self, params: GenomeToolParams, context: RunContext) -> list[str]:
        return _check_genomes(context, params.vtools)

    def label(self, params: GenomeToolParams) -> str:
        return ", ".join(params.vtools) if params.vtools else "all vtools"

    def inputs(self, params: GenomeToolParams, context: RunContext) -> list[Path]:
        return _genome_dirs(context, params.vtools)

    def writes(self, params: GenomeToolParams, context: RunContext) -> list[Path]:
        return [
            d / n
            for d in self._dirs(context)
            for n in _virus_identification_tool_names(context, params.vtools)
        ]

    def run(self, params: GenomeToolParams, context: RunContext) -> list[Path] | None:
        run_snakemake(
            context,
            "all_phanotate",
            params.vtools,
            params.samples,
            params.cores,
            dry_run=params.dry_run,
        )
        return None if params.dry_run else [d for d in self.writes(params, context) if d.is_dir()]


@dataclass(frozen=True)
class Phabox2Params:
    tasks: list[Literal["cherry", "phagcn", "phavip"]] = param(
        default_factory=lambda: list(PHABOX2_TASKS),
        help="cherry/phagcn run on genomes, phavip on PHANOTATE ORFs",
    )
    vtools: list[str] | None = param(None, help=VIRUS_IDENTIFICATION_TOOLS_HELP)
    samples: list[str] | None = param(None, help=SAMPLES_HELP)
    cores: int | None = param(None, help=CORES_HELP)
    dry_run: bool = param(False, help=DRY_RUN_HELP)


@register
class Phabox2(Stage):
    name = "tools.phabox2"
    summary = "PhaBOX2 host (cherry), taxonomy (phagcn) and gene function (phavip)"
    Params = Phabox2Params

    def resolve(self, params, context: RunContext):
        return _resolve_virus_identification_tools(params, context)

    def validate(self, params: Phabox2Params, context: RunContext) -> list[str]:
        errors = _check_virus_identification_tools(context, params.vtools)
        if not params.tasks:
            errors.append("tasks: give at least one of cherry, phagcn, phavip")
        tags = {modality["tool"] for modality in context.modalities.values()}
        errors += [
            f"no [modalities] entry with tool = {t!r} (its filename tag)"
            for t in params.tasks
            if t not in tags
        ]
        keys = [
            ("phabox2", "path"),
            ("databases", "phabox2"),
            ("threads", "phabox2"),
        ]
        return errors + _require(context, keys)

    def check_inputs(self, params: Phabox2Params, context: RunContext) -> list[str]:
        errors = []
        if {"cherry", "phagcn"} & set(params.tasks):
            errors += _check_genomes(context, params.vtools)
        if "phavip" in params.tasks:
            orfs = context.section_path("phanotate")
            errors += [
                f"no PHANOTATE ORFs in {orfs / n} -- run tools.phanotate first"
                for n in _virus_identification_tool_names(context, params.vtools)
                if not (orfs / n).is_dir() or not any((orfs / n).glob("*.fna"))
            ]
        return errors

    def label(self, params: Phabox2Params) -> str:
        return " + ".join(params.tasks)

    def inputs(self, params: Phabox2Params, context: RunContext) -> list[Path]:
        dirs = []
        if {"cherry", "phagcn"} & set(params.tasks):
            dirs += _genome_dirs(context, params.vtools)
        if "phavip" in params.tasks:
            dirs += [
                context.section_path("phanotate") / n
                for n in _virus_identification_tool_names(context, params.vtools)
            ]
        return dirs

    def writes(self, params: Phabox2Params, context: RunContext) -> list[Path]:
        root = context.section_path("phabox2")
        return [
            root / t / n
            for t in params.tasks
            for n in _virus_identification_tool_names(context, params.vtools)
        ]

    def run(self, params: Phabox2Params, context: RunContext) -> list[Path] | None:
        targets = [f"all_phabox2_{task}" for task in params.tasks]
        run_snakemake(
            context, targets, params.vtools, params.samples, params.cores, dry_run=params.dry_run
        )
        return None if params.dry_run else [d for d in self.writes(params, context) if d.is_dir()]


# installation


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

    def validate(self, params: SetupParams, context: RunContext) -> list[str]:
        return _require(
            context, [("workflow", "snakemake"), ("workflow", "snakefile"), ("workflow", "workdir")]
        )

    def label(self, params: SetupParams) -> str:
        return ", ".join(params.tools)

    def inputs(self, params: SetupParams, context: RunContext) -> list[Path]:
        return []

    def writes(self, params: SetupParams, context: RunContext) -> list[Path]:
        return [context.resolve(_config_value(context, "workflow", "workdir")) / "setup"]

    def run(self, params: SetupParams, context: RunContext) -> list[Path] | None:
        setup_dir = context.resolve(_config_value(context, "workflow", "workdir")) / "setup"
        # The markers are cheap to recreate; always recheck that tools start.
        for tool in params.tools:
            (setup_dir / f"{tool}.ok").unlink(missing_ok=True)
        run_snakemake(context, [f"setup_{t}" for t in params.tools], dry_run=params.dry_run)
        return None if params.dry_run else [setup_dir]
