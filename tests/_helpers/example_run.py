"""
Running the synthetic example pipeline (``phantom example`` + ``runs/example.toml``)
through the command line, for the regression and end-to-end tests.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from phantom.config.loader import ConfigLoader
from phantom.example import write_example
from phantom.runner.cli import main

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE_RUN_FILE = REPOSITORY / "runs" / "example.toml"


@contextmanager
def example_workspace(root: Path):
    """Writes the example workspace into ``root``, runs the whole example run
    file on it and yields (config path, exit code). `phantom --config`
    activates the example's configuration for the process; it is undone after."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ConfigLoader, "ACTIVE_PATH", ConfigLoader.ACTIVE_PATH)
        patch.setattr(ConfigLoader, "DATA_ROOT", ConfigLoader.DATA_ROOT)
        config = write_example(root)
        exit_code = main(["--config", str(config), "run", str(EXAMPLE_RUN_FILE)])
        yield config, exit_code
