"""Project scaffolding; templates shipped as package data (fix #11)."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

_JUNK_DIRS = {".ruff_cache", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv"}
_JUNK_SUFFIXES = {".pyc", ".pyo"}


def template_dir() -> Path:
    """Templates shipped inside the package (wheel-safe)."""
    return Path(files("lazy107").joinpath("templates/default"))


def scaffold(target: Path, project_name: str) -> Path:
    """Copy templates/default into *target*; substitute {project} in text files."""
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"cannot scaffold into a non-empty directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(template_dir().rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(template_dir())
        if any(part in _JUNK_DIRS for part in rel.parts) or src.suffix in _JUNK_SUFFIXES:
            continue
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            src.read_text(encoding="utf-8").replace("{project}", project_name),
            encoding="utf-8",
        )
    return target
