"""The license travels with the code: the GPL text is in the repository and
the program's main file carries the notice the GPL asks for (copyright,
GPL-3.0-or-later, no warranty, where the full text is)."""

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
MAIN_FILE = REPOSITORY / "src" / "phantom" / "__main__.py"


def test_the_license_text_is_in_the_repository():
    text = (REPOSITORY / "LICENSE").read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in text and "Version 3, 29 June 2007" in text


def test_the_main_file_carries_the_gpl_notice():
    notice = " ".join(
        line.lstrip("# ").strip()
        for line in MAIN_FILE.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    )
    for part in (
        "Copyright (C)",
        "either version 3 of the License, or (at your option) any later version",
        "WITHOUT ANY WARRANTY",
        "<https://www.gnu.org/licenses/>",
    ):
        assert part in notice, f"{MAIN_FILE.name}: the GPL notice lacks {part!r}"
