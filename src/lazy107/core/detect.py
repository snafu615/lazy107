"""Entry-point discovery (train.py > main.py > other .py > notebooks)."""

from __future__ import annotations

import os
from pathlib import Path

ENTRY_PRIORITY = ("train.py", "main.py")


def _ignore_walk_error(_error: OSError) -> None:
    """Skip directories that cannot be listed (permissions, broken mounts)."""


def detect_entries(project_root: Path) -> list[Path]:
    if not project_root.exists():
        return []
    py: list[Path] = []
    nb: list[Path] = []
    for root, _dirs, names in os.walk(project_root, onerror=_ignore_walk_error):
        for name in names:
            path = Path(root) / name
            if not path.is_file():
                continue
            if path.suffix == ".py":
                py.append(path)
            elif path.suffix == ".ipynb":
                nb.append(path)
    rank = {name: i for i, name in enumerate(ENTRY_PRIORITY)}
    preferred = sorted(
        (p for p in py if p.name in rank),
        key=lambda p: (rank[p.name], str(p)),
    )
    other = sorted(p for p in py if p.name not in rank)
    return preferred + other + sorted(nb)


def recommend_entry(project_root: Path) -> Path | None:
    entries = detect_entries(project_root)
    return entries[0] if entries else None
