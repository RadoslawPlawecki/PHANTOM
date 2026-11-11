"""Helpers shared by stage adapters."""

from __future__ import annotations

from pathlib import Path

from phantom.preprocessing.metadata.mapping import MetadataManager
from phantom.runner.stage import RunContext


def check_version(version: str, ctx: RunContext) -> list[str]:
    """Declared in config.toml, or created by an earlier step of this run."""
    if ctx.version_known(version):
        return []
    versions = ctx.feature_versions()
    return [f"version {version!r} not in config.toml (available: {', '.join(versions)})"]


def check_vtool(vtool: str, ctx: RunContext) -> list[str]:
    if vtool in ctx.vtools:
        return []
    return [f"vtool {vtool!r} not in [tools] (available: {', '.join(ctx.vtools)})"]


def output_dir(out_dir: str | None, ctx: RunContext, *default_parts: str) -> Path:
    """An explicit ``out_dir`` parameter, else ``<[results]>/<default_parts...>``."""
    if out_dir:
        return ctx.resolve(out_dir)
    return ctx.section_path("results").joinpath(*default_parts)


def metadata_path(ctx: RunContext) -> Path:
    return MetadataManager.path_from_config(ctx.config)


def check_metadata(ctx: RunContext) -> list[str]:
    path = metadata_path(ctx)
    return [] if path.is_file() else [f"metadata with class labels not found: {path}"]
