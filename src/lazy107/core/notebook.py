"""Notebook -> Python conversion (nbconvert imported lazily)."""

from __future__ import annotations

import re
from pathlib import Path

# Interactive-display code has no place in a batch job: drop IPython/ipywidgets
# imports, notebook display calls and matplotlib show() lines.
_STRIP_PATTERNS = (
    re.compile(
        r"^\s*(?:from\s+(?:IPython|ipywidgets)\b.*?import|import\s+(?:IPython|ipywidgets)\b).*$",
        re.MULTILINE,
    ),
    re.compile(r"^\s*(?:clear_output|display)\s*\(.*$", re.MULTILINE),
    re.compile(r"^\s*plt\.show\s*\(.*$", re.MULTILINE),
)


def _backup(path: Path) -> None:
    """Move an existing output aside: test.py -> test.py.old, .old2, ..."""
    candidate = Path(str(path) + ".old")
    n = 2
    while candidate.exists():
        candidate = Path(f"{path}.old{n}")
        n += 1
    path.rename(candidate)


def _collapse_blank_runs(body: str) -> str:
    """Rstrip every line and collapse runs of blank lines to a single blank."""
    kept: list[str] = []
    prev_blank = False
    for line in body.splitlines():
        line = line.rstrip()
        if line:
            kept.append(line)
            prev_blank = False
        elif not prev_blank:
            kept.append("")
            prev_blank = True
    return "\n".join(kept) + "\n"


def _extract_code_cells(notebook_path: Path) -> str:
    """Extract code cells straight from the notebook JSON (nbconvert-free)."""
    import json

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    chunks = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source and not source.endswith("\n"):
            source += "\n"
        chunks.append(source)
    return "".join(chunks)


def convert_notebook(notebook_path: Path, output_path: Path) -> Path:
    """Convert a notebook to a plain .py script, stripping interactive code.

    Backs up an existing output file before overwriting.

    Uses nbconvert when installed; otherwise falls back to extracting the
    code cells directly from the notebook JSON (nbconvert is heavy and
    optional, so the CLI must not hard-fail without it).
    """
    try:
        from nbconvert.exporters import PythonExporter  # lazy: heavy dependency
    except ImportError:
        body = _extract_code_cells(notebook_path)
    else:
        body, _ = PythonExporter().from_filename(str(notebook_path))
    for pattern in _STRIP_PATTERNS:
        body = pattern.sub("", body)
    body = _collapse_blank_runs(body)

    if output_path.exists():
        _backup(output_path)
    output_path.write_text(body, encoding="utf-8")
    return output_path
