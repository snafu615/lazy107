"""Large-file discovery → GUI upload checklist (print-only; no transfer)."""

from __future__ import annotations

import os
from pathlib import Path

# Keep in sync with templates/default/.gitignore (enforced by test_transfer).
LARGE_SUFFIXES = frozenset({
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5", ".hdf5",
    ".npz", ".npy", ".tar", ".zip", ".gz", ".bin", ".db", ".sqlite", ".parquet",
})

LARGE_DIRS = ("data", "datasets", "checkpoints", "outputs", "logs", "models")

_ARCHIVE_SUFFIX = "-data.tar.gz"


def _ignore_walk_error(_error: OSError) -> None:
    """Skip directories that cannot be listed (permissions, broken mounts)."""


def large_files(project_root: Path) -> list[Path]:
    """Project-relative paths of files matching large-data patterns (sorted)."""
    found: set[Path] = set()
    for root, _dirs, names in os.walk(project_root, onerror=_ignore_walk_error):
        for name in names:
            path = Path(root) / name
            if not path.is_file():
                continue
            rel = path.relative_to(project_root)
            if rel.name.endswith(_ARCHIVE_SUFFIX):
                continue  # never include the checklist's own archive
            if rel.parts[0] in LARGE_DIRS or path.suffix.lower() in LARGE_SUFFIXES:
                found.add(rel)
    return sorted(found)


def checklist(project_root: Path) -> list[str]:
    """Printed recipe: tar → GUI upload → extract → sha256."""
    files = large_files(project_root)
    if not files:
        return ["No large files found; nothing to transfer."]
    archive = f"{project_root.name}{_ARCHIVE_SUFFIX}"
    tar = f"tar -czf {archive} " + " ".join(f'"{f.as_posix()}"' for f in files)
    return [
        f"Large files ({len(files)}):",
        *[f"  {f.as_posix()}" for f in files],
        "",
        "# 1. Bundle locally (outside git):",
        f"  {tar}",
        "# 2. Upload via GUI (SCOW file manager / Xftp) to ~/transfer/ on the login node",
        "# 3. Extract on the cluster:",
        f"  mkdir -p ~/transfer && tar -xzf {archive} -C .",
        "# 4. Verify integrity (compare with local sha256):",
        f"  sha256sum {archive}",
    ]
