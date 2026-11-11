"""
The ``phantom`` command. Fully non-interactive: what to compute comes from a
run file, never from prompts.

    phantom stages [STAGE]            list stages / describe one
    phantom new STAGE [-o FILE]       write a run-file template
    phantom validate RUNFILE          check a run file without running it
    phantom plan RUNFILE              show what would run and what is up to date
    phantom run RUNFILE               run it (skips up-to-date jobs)

``validate``/``plan``/``run`` accept ``--set key=value`` (repeatable) to
override a parameter without editing the file; ``plan``/``run`` also accept
``--only``, ``--from`` and ``--force``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import phantom.stages  # noqa: F401 -- registers every stage
from phantom.runner.loader import RunFileError, apply_overrides, build_jobs, load_run_file
from phantom.runner.planner import JobFailed, execute, format_plan, plan, resolve_from
from phantom.runner.provenance import phantom_version
from phantom.runner.stage import REGISTRY, RunContext, get_stage
from phantom.runner.stamps import StampStore
from phantom.runner.template import describe_stage, render_template, stages_markdown

EXIT_OK, EXIT_INVALID, EXIT_FAILED = 0, 1, 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phantom",
        description="PHANTOM -- phageome-based disease classification. Runs are described "
        "by run files (runs/*.toml); see `phantom stages` and `phantom new`.",
    )
    parser.add_argument("--version", action="version", version=f"phantom {phantom_version()}")
    parser.add_argument(
        "--config", type=Path, default=None, help="config.toml to use (default: the repo's)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stages", help="list stages, or describe one")
    p.add_argument("stage", nargs="?")
    p.add_argument("--markdown", action="store_true", help="every stage as a Markdown page")

    p = sub.add_parser("new", help="write a run-file template for a stage")
    p.add_argument("stage")
    p.add_argument("-o", "--output", type=Path, help="write to FILE instead of stdout")

    for name, text in (
        ("validate", "check a run file without running anything"),
        ("plan", "show which jobs would run and which are up to date"),
        ("run", "run a run file"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("runfile", type=Path)
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a parameter (repeatable); KEY or STAGE.KEY",
        )
        if name in ("plan", "run"):
            p.add_argument(
                "--only",
                type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                help="comma-separated stage names or step numbers to run",
            )
            p.add_argument("--from", dest="start", help="start at this step number or stage")
            p.add_argument("--force", action="store_true", help="rerun even if up to date")
    return parser


def _cmd_stages(args, ctx: RunContext) -> int:
    if args.markdown:
        print(stages_markdown(list(REGISTRY.values()), ctx))
        return EXIT_OK
    if args.stage:
        print(describe_stage(get_stage(args.stage), ctx))
        return EXIT_OK
    width = max(len(n) for n in REGISTRY)
    for name in sorted(REGISTRY):
        print(f"{name.ljust(width)}  {REGISTRY[name].summary}")
    return EXIT_OK


def _cmd_new(args, ctx: RunContext) -> int:
    text = render_template(get_stage(args.stage), ctx)
    if args.output:
        if args.output.exists():
            print(f"[ERROR] {args.output} already exists; not overwriting.", file=sys.stderr)
            return EXIT_INVALID
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text, end="")
    return EXIT_OK


def _load(args, ctx: RunContext):
    run = apply_overrides(load_run_file(args.runfile), args.overrides)
    return run, build_jobs(run, ctx)


def _cmd_validate(args, ctx: RunContext) -> int:
    _, jobs = _load(args, ctx)
    # Plan as if everything ran, so the input data of every job that does not
    # depend on an earlier one is checked too (dependent jobs: when they run).
    planned = plan(jobs, ctx, StampStore(ctx.section_path("stamps")), force=True)
    deferred = sum(pj.deferred for pj in planned)
    steps = len({j.step.index for j in jobs})
    note = f"; {deferred} read inputs an earlier job produces" if deferred else ""
    print(f"[OK] {args.runfile}: {len(jobs)} job(s) in {steps} step(s){note}.")
    return EXIT_OK


def _plan(args, ctx: RunContext):
    run, jobs = _load(args, ctx)
    store = StampStore(ctx.section_path("stamps"))
    planned = plan(
        jobs,
        ctx,
        store,
        force=args.force,
        only=args.only,
        start=resolve_from(run, args.start),
    )
    return run, store, planned


def _cmd_plan(args, ctx: RunContext) -> int:
    _, _, planned = _plan(args, ctx)
    print(format_plan(planned))
    return EXIT_OK


def _cmd_run(args, ctx: RunContext) -> int:
    run, store, planned = _plan(args, ctx)
    print(format_plan(planned))
    try:
        produced = execute(planned, ctx, store, run)
    except JobFailed as e:
        print(f"\n[FAILED] {e}", file=sys.stderr)
        print(
            "Later jobs were not run. Fix the problem and rerun; finished jobs are skipped.",
            file=sys.stderr,
        )
        return EXIT_FAILED
    print(f"\n[SUCCESS] {len(produced)} output(s) written.")
    return EXIT_OK


COMMANDS = {
    "stages": _cmd_stages,
    "new": _cmd_new,
    "validate": _cmd_validate,
    "plan": _cmd_plan,
    "run": _cmd_run,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        ctx = RunContext.load(args.config)
        return COMMANDS[args.command](args, ctx)
    except RunFileError as e:
        print(f"[INVALID] {getattr(args, 'runfile', '')}:", file=sys.stderr)
        for msg in e.errors:
            print(f"  - {msg}", file=sys.stderr)
        return EXIT_INVALID
    except (KeyError, ValueError) as e:
        print(f"[ERROR] {e.args[0] if e.args else e}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
