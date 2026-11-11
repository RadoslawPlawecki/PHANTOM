"""
Declaring and checking stage parameters.

A stage's ``Params`` is a frozen dataclass whose fields are declared with
:func:`param`. The field's type annotation is the schema: it drives
validation of run files, the ``phantom new`` template and the stamp hash, so
there is exactly one place that says what a stage accepts.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import MISSING, Field
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin


def param(
    default: Any = MISSING,
    *,
    default_factory: Any = MISSING,
    help: str = "",
    grid: bool = False,
) -> Any:
    """
    A stage parameter. ``grid=True`` lets a run file give a *list* for this
    (scalar) field, which expands into one job per value -- several grid
    fields expand into their cartesian product.
    """
    metadata = {"help": help, "grid": grid}
    if default_factory is not MISSING:
        return dataclasses.field(default_factory=default_factory, metadata=metadata)
    return dataclasses.field(default=default, metadata=metadata)


def fields(params_cls: type) -> list[Field]:
    return list(dataclasses.fields(params_cls))


def type_hints(params_cls: type) -> dict[str, Any]:
    return typing.get_type_hints(params_cls)


def is_grid(f: Field) -> bool:
    return bool(f.metadata.get("grid", False))


def help_text(f: Field) -> str:
    return str(f.metadata.get("help", ""))


def has_static_default(f: Field) -> bool:
    return f.default is not MISSING or f.default_factory is not MISSING


def static_default(f: Field) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    raise KeyError(f.name)


def literal_choices(tp: Any) -> tuple[Any, ...] | None:
    """The allowed values of a ``Literal[...]`` (or ``list[Literal[...]]``)."""
    if get_origin(tp) is Literal:
        return get_args(tp)
    if get_origin(tp) is list:
        (inner,) = get_args(tp) or (Any,)
        return literal_choices(inner)
    if _is_union(tp):
        for arg in get_args(tp):
            choices = literal_choices(arg)
            if choices is not None:
                return choices
    return None


def describe(tp: Any) -> str:
    """Human-readable type, for error messages and templates."""
    origin = get_origin(tp)
    if origin is Literal:
        return "one of " + ", ".join(repr(a) for a in get_args(tp))
    if _is_union(tp):
        return " or ".join(describe(a) for a in get_args(tp))
    if origin is list:
        (inner,) = get_args(tp) or (Any,)
        return f"list of ({describe(inner)})"
    if origin is dict:
        k, v = get_args(tp) or (Any, Any)
        return f"table of {describe(k)} -> {describe(v)}"
    if tp is type(None):
        return "nothing"
    if tp in (str, Path):
        return "string"
    return getattr(tp, "__name__", str(tp))


def check(value: Any, tp: Any) -> bool:
    """Does ``value`` (as parsed from TOML) conform to annotation ``tp``?"""
    origin = get_origin(tp)
    if tp is Any:
        return True
    if origin is Literal:
        return value in get_args(tp)
    if _is_union(tp):
        return any(check(value, arg) for arg in get_args(tp))
    if tp is type(None):
        return value is None
    if origin is list:
        (inner,) = get_args(tp) or (Any,)
        return isinstance(value, list) and all(check(v, inner) for v in value)
    if origin is dict:
        k, v = get_args(tp) or (Any, Any)
        return isinstance(value, dict) and all(
            check(key, k) and check(val, v) for key, val in value.items()
        )
    if tp is bool:
        return isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    if tp in (str, Path):
        return isinstance(value, str)
    return isinstance(value, tp)


def _is_union(tp: Any) -> bool:
    return get_origin(tp) in (Union, types.UnionType)
