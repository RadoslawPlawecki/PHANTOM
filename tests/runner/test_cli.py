from __future__ import annotations

import tomllib

from phantom.runner.cli import EXIT_FAILED, EXIT_INVALID, EXIT_OK, main


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_stages_lists_registered(toy, tmp_workspace, capsys):
    assert main(["--config", str(tmp_workspace.config_path), "stages"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "classification.single" in out
    assert "test.echo" in out


def test_new_writes_template_and_refuses_overwrite(toy, tmp_workspace, tmp_path, capsys):
    cfg = ["--config", str(tmp_workspace.config_path)]
    target = tmp_path / "runs" / "echo.toml"
    assert main([*cfg, "new", "test.echo", "-o", str(target)]) == EXIT_OK
    doc = tomllib.loads(target.read_text(encoding="utf-8"))
    assert doc["steps"][0]["stage"] == "test.echo"
    # The template is immediately valid (the required str placeholder is "").
    assert main([*cfg, "validate", str(target)]) == EXIT_OK
    assert main([*cfg, "new", "test.echo", "-o", str(target)]) == EXIT_INVALID


def test_validate_reports_errors(toy, tmp_workspace, tmp_path, capsys):
    f = _write(tmp_path / "r.toml", '[[steps]]\nstage = "test.echo"\nflag = "yes"\n')
    assert main(["--config", str(tmp_workspace.config_path), "validate", str(f)]) == EXIT_INVALID
    err = capsys.readouterr().err
    assert "missing required parameter 'name'" in err
    assert "flag: expected bool" in err


def test_run_then_plan_shows_up_to_date(toy, tmp_workspace, tmp_path, capsys):
    cfg = ["--config", str(tmp_workspace.config_path)]
    f = _write(tmp_path / "r.toml", '[[steps]]\nstage = "test.echo"\nname = "a"\n')
    assert main([*cfg, "run", str(f)]) == EXIT_OK
    capsys.readouterr()
    assert main([*cfg, "plan", str(f)]) == EXIT_OK
    assert "0 job(s) to run, 1 up to date." in capsys.readouterr().out
    assert main([*cfg, "plan", str(f), "--set", "n=2"]) == EXIT_OK
    assert "1 job(s) to run" in capsys.readouterr().out


def test_run_failure_exit_code(toy, tmp_workspace, tmp_path, capsys):
    echo, _, _ = toy
    echo.fail_on = "a"
    f = _write(tmp_path / "r.toml", '[[steps]]\nstage = "test.echo"\nname = "a"\n')
    assert main(["--config", str(tmp_workspace.config_path), "run", str(f)]) == EXIT_FAILED
    assert "[FAILED]" in capsys.readouterr().err


def test_validate_checks_input_data_not_only_the_file(toy, tmp_workspace, tmp_path, capsys):
    cfg = ["--config", str(tmp_workspace.config_path)]
    alone = _write(tmp_path / "c.toml", '[[steps]]\nstage = "test.consume"\n')
    assert main([*cfg, "validate", str(alone)]) == EXIT_INVALID
    assert "nothing to consume" in capsys.readouterr().err
    # After a producer in the same file the check waits for the producer.
    chain = _write(
        tmp_path / "p.toml",
        '[[steps]]\nstage = "test.echo"\nname = "a"\n\n[[steps]]\nstage = "test.consume"\n',
    )
    assert main([*cfg, "validate", str(chain)]) == EXIT_OK
    assert "1 read inputs an earlier job produces" in capsys.readouterr().out


def test_stages_markdown_lists_every_stage(toy, tmp_workspace, capsys):
    assert main(["--config", str(tmp_workspace.config_path), "stages", "--markdown"]) == EXIT_OK
    page = capsys.readouterr().out
    from phantom.runner.stage import REGISTRY

    assert all(f"## `{name}`" in page for name in REGISTRY)
    assert "| `name` | string (**grid**) | **required** |" in page
