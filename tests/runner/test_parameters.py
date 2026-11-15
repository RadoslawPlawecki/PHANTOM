"""Stage parameters: type checking of run-file values against the Params annotations."""

from __future__ import annotations

from typing import Literal

import pytest

from phantom.runner import parameters


@pytest.mark.parametrize(
    ("value", "tp", "ok"),
    [
        ("a", str, True),
        (1, str, False),
        (1, int, True),
        (True, int, False),  # bool is not an int here
        (1, float, True),
        (1.5, float, True),
        (True, bool, True),
        ("no", bool, False),
        ("rf", Literal["rf", "xgb"], True),
        ("svm", Literal["rf", "xgb"], False),
        (["rf"], list[Literal["rf", "xgb"]], True),
        (["rf", "svm"], list[Literal["rf", "xgb"]], False),
        ("rf", list[str], False),
        (None, str | None, True),
        ("x", str | None, True),
        (3, str | None, False),
        ({"a": "b"}, dict[str, str], True),
        ({"a": 1}, dict[str, str], False),
    ],
)
def test_check(value, tp, ok):
    assert parameters.check(value, tp) is ok


def test_describe_and_choices():
    tp = list[Literal["loocv", "rcv"]]
    assert parameters.describe(tp) == "list of (one of 'loocv', 'rcv')"
    assert parameters.literal_choices(tp) == ("loocv", "rcv")
    assert parameters.literal_choices(str) is None
    assert parameters.describe(str | None) == "string or nothing"
