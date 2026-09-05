"""Fixtures shared across the whole test suite."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _neutralize_lazy107_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop LAZY107_* variables so tests never inherit a developer's config."""
    for key in list(os.environ):
        if key.startswith("LAZY107_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _isolate_global_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the per-user global config at a per-test temp path."""
    monkeypatch.setattr("lazy107.manifest.global_config_path", lambda: tmp_path / "global-config.toml")


@pytest.fixture
def blocked_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A directory whose scandir raises PermissionError (simulates an unreadable dir)."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    real_scandir = os.scandir

    def scandir(path="."):
        if Path(path) == blocked:
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    return blocked
