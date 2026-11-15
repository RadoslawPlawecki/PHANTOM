"""
The example pipeline reproduces its reference results: every selected output
(golden_files.SELECTED) of a fresh run equals the reviewed copy in golden/,
numbers within a relative tolerance of 1e-6. One test per file, so a failure
names the step that changed.

After an intended change of results: python tests/regression/update_golden.py,
then review the diff of tests/regression/golden/ before committing it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._helpers.example_run import example_workspace
from tests.regression.golden_files import (
    FUNCTION_SUMMARY,
    comparable_text,
    differences,
    function_summary,
    selected_outputs,
)

GOLDEN = Path(__file__).parent / "golden"
GOLDEN_FILES = sorted(str(p.relative_to(GOLDEN)) for p in GOLDEN.rglob("*") if p.is_file())


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    root = tmp_path_factory.mktemp("example")
    with example_workspace(root) as (_, exit_code):
        assert exit_code == 0, "the example pipeline failed"
        yield root


def test_the_reference_results_exist():
    assert GOLDEN_FILES, "golden/ is empty: run tests/regression/update_golden.py"


def test_the_pipeline_writes_exactly_the_reference_outputs(example):
    assert sorted([*selected_outputs(example), FUNCTION_SUMMARY]) == GOLDEN_FILES


@pytest.mark.parametrize("name", GOLDEN_FILES)
def test_output_matches_the_reference(example, name, tmp_path):
    comparable = tmp_path / Path(name).name
    if name == FUNCTION_SUMMARY:
        function_summary(example).to_csv(comparable, index=False)
    else:
        produced = selected_outputs(example).get(name)
        assert produced is not None, f"{name} was not produced"
        comparable.write_text(comparable_text(produced, example), encoding="utf-8")
    problems = differences(GOLDEN / name, comparable)
    assert not problems, f"{name} changed:\n  " + "\n  ".join(problems[:20])
