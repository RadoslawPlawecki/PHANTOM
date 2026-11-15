"""
Re-records the reference results in tests/regression/golden/ from a fresh run
of the example pipeline. Only after an intended change of results -- then
review `git diff tests/regression/golden` before committing.

    python tests/regression/update_golden.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from tests._helpers.example_run import example_workspace  # noqa: E402
from tests.regression.golden_files import (  # noqa: E402
    FUNCTION_SUMMARY,
    comparable_text,
    function_summary,
    selected_outputs,
)

GOLDEN = Path(__file__).parent / "golden"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "example"
        with example_workspace(root) as (_, exit_code):
            if exit_code != 0:
                print("the example pipeline failed; golden files left unchanged")
                return exit_code
            shutil.rmtree(GOLDEN, ignore_errors=True)
            outputs = selected_outputs(root)
            for name, path in outputs.items():
                target = GOLDEN / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(comparable_text(path, root), encoding="utf-8")
            (GOLDEN / FUNCTION_SUMMARY).parent.mkdir(parents=True, exist_ok=True)
            function_summary(root).to_csv(GOLDEN / FUNCTION_SUMMARY, index=False)
    print(f"wrote {len(outputs) + 1} reference files to {GOLDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
