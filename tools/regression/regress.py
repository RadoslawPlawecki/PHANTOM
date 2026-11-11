"""
Legacy-vs-new regression checks for the menu -> run-file migration.

    python tools/regression/regress.py list
    python tools/regression/regress.py baseline CASE|PREFIX ... [--repeat]
    python tools/regression/regress.py check CASE|PREFIX ...

``check`` runs a case's ``phantom`` step and fails unless every output
matches the recorded baseline (``<workspace>/.phantom/regression/baselines``).
``baseline`` records the current output as the new reference (after an
intended change of results); ``--repeat`` refuses a nondeterministic one. A PREFIX such as
``features.`` selects every case whose name starts with it. See harness.py.
"""

from __future__ import annotations

import argparse
import sys
import time

import harness
from cases import BY_NAME, CASES


def _select(names: list[str]) -> list[harness.Case]:
    selected = []
    for name in names:
        hits = [c for c in CASES if c.name == name or c.name.startswith(name)]
        if not hits:
            raise SystemExit(f"[ERROR] no case matches {name!r}; see `regress.py list`")
        selected += [c for c in hits if c not in selected]
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    b = sub.add_parser("baseline")
    b.add_argument("cases", nargs="+")
    b.add_argument("--repeat", action="store_true", help="verify the legacy path is deterministic")
    c = sub.add_parser("check")
    c.add_argument("cases", nargs="+")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for case in CASES:
            state = "baseline" if harness.baseline_path(case).is_file() else "no baseline"
            print(f"{case.name:40s} {state}")
        return 0

    failures = []
    for case in _select(args.cases):
        print(f"\n##### {args.cmd} {case.name}", flush=True)
        t0 = time.time()
        if args.cmd == "baseline":
            harness.record_baseline(case, repeat=args.repeat)
        elif not harness.check(case):
            failures.append(case.name)
        print(f"##### {case.name}: {time.time() - t0:.0f}s", flush=True)
    if failures:
        print(f"\n[FAILED] {len(failures)} case(s): {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["BY_NAME", "main"]
