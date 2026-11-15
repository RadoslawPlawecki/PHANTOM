"""
Names of the two classes -- label 0 and label 1 -- as figures and file names
show them. The labels themselves are always 0/1 (from the class folders of
the raw folder, recorded in the metadata's ``label`` column); ``[classes]``
in config.toml only names them, e.g. ``control = "Healthy"``,
``case = "Allergy"`` -- and the class folders may carry these names.
Default: "Control", "Case".
"""

from __future__ import annotations

import re

from phantom.config.loader import ConfigLoader

DEFAULTS = {"control": "Control", "case": "Case"}


def class_names(config: dict | None = None) -> dict[int, str]:
    """{0: control name, 1: case name}, from ``config`` (default: the active configuration)."""
    if config is None:
        config = ConfigLoader().load()
    names = {**DEFAULTS, **config.get("classes", {})}
    return {0: names["control"], 1: names["case"]}


def slug(name: str) -> str:
    """A class name as a file-name part: "Allergy" -> "allergy", "Crohn's disease" -> "crohn_s_disease"."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
