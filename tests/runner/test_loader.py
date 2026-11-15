"""Run files -> jobs: parsing, --set overrides, grids, and every error at once."""

from __future__ import annotations

import pytest

from phantom.runner.loader import (
    RunFileError,
    apply_overrides,
    build_jobs,
    load_run_file,
    parse_override,
    parse_run_doc,
)


def _jobs(doc, context, overrides=()):
    return build_jobs(apply_overrides(parse_run_doc(doc), list(overrides)), context)


def test_grid_expands_lists_into_jobs_in_order(toy, context):
    jobs = _jobs({"steps": [{"stage": "test.echo", "name": "a", "n": [1, 2, 3]}]}, context)
    assert [j.params.n for j in jobs] == [1, 2, 3]
    assert all(j.params.name == "a" for j in jobs)


def test_non_grid_list_field_is_not_expanded(toy, context):
    (job,) = _jobs({"steps": [{"stage": "test.echo", "name": "a", "tags": ["a", "b"]}]}, context)
    assert job.params.tags == ["a", "b"]


def test_defaults_are_filled(toy, context):
    (job,) = _jobs({"steps": [{"stage": "test.echo", "name": "a"}]}, context)
    assert (job.params.n, job.params.tags, job.params.flag, job.params.src) == (1, [], False, None)


def test_all_errors_in_a_step_are_reported_together(toy, context):
    doc = {
        "steps": [
            {"stage": "test.ech"},
            {"stage": "test.echo", "names": "a", "n": [1, "two"], "flag": "yes", "tags": ["c"]},
        ]
    }
    with pytest.raises(RunFileError) as exc:
        _jobs(doc, context)
    text = "\n".join(exc.value.errors)
    assert "did you mean 'test.echo'" in text
    assert "unknown parameter 'names' -- did you mean 'name'?" in text
    assert "missing required parameter 'name'" in text
    assert "n: expected int, got 'two'" in text
    assert "flag: expected bool, got 'yes'" in text
    assert "tags: expected list of (one of 'a', 'b'), got ['c']" in text


def test_empty_grid_list_is_an_error(toy, context):
    with pytest.raises(RunFileError, match="n: empty list"):
        _jobs({"steps": [{"stage": "test.echo", "name": "a", "n": []}]}, context)


@pytest.mark.parametrize(
    "doc",
    [
        {},
        {"steps": []},
        {"steps": [{"name": "a"}]},
        {"steps": [{"stage": "test.echo"}], "extra": {}},
    ],
)
def test_malformed_documents(doc):
    with pytest.raises(RunFileError):
        parse_run_doc(doc)


def test_provenance_table_is_accepted(toy, context):
    doc = {"provenance": {"git_commit": "abc"}, "steps": [{"stage": "test.echo", "name": "a"}]}
    assert len(_jobs(doc, context)) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("n=3", (None, "n", 3)),
        ("flag=true", (None, "flag", True)),
        ('tags=["a","b"]', (None, "tags", ["a", "b"])),
        ("name=rf", (None, "name", "rf")),  # bare word -> string
        ("test.echo.name=x", ("test.echo", "name", "x")),
    ],
)
def test_parse_override(text, expected):
    assert parse_override(text) == expected


def test_override_applies_to_matching_steps_only(toy, context):
    doc = {"steps": [{"stage": "test.echo", "name": "a"}, {"stage": "test.consume"}]}
    jobs = _jobs(doc, context, ["n=[4,5]", "test.consume.tag=z"])
    assert [j.params.n for j in jobs if j.stage.name == "test.echo"] == [4, 5]
    assert [j.params.tag for j in jobs if j.stage.name == "test.consume"] == ["z"]


def test_override_without_target_is_an_error(toy, context):
    with pytest.raises(RunFileError, match="no parameter 'nope'"):
        _jobs({"steps": [{"stage": "test.echo", "name": "a"}]}, context, ["nope=1"])


def test_load_run_file_errors(tmp_path):
    with pytest.raises(RunFileError, match="not found"):
        load_run_file(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[[steps]\n", encoding="utf-8")
    with pytest.raises(RunFileError, match="invalid TOML"):
        load_run_file(bad)
