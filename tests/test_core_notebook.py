"""Tests for notebook conversion."""

import json
import sys
from pathlib import Path

from lazy107.core.notebook import convert_notebook


def _notebook_doc(cells: list[dict]) -> dict:
    for i, cell in enumerate(cells):
        cell.setdefault("id", f"nb-cell-{i}")
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _write_demo_notebook(path: Path) -> None:
    cells = [
        {
            "cell_type": "code",
            "execution_count": 10,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "from IPython.display import clear_output\n",
                "import matplotlib.pyplot as plt\n",
                "from ipywidgets import Dropdown\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 11,
            "metadata": {},
            "outputs": [],
            "source": ["x = torch.randn(10)\n", "plt.plot(x)\n", "clear_output(wait=True)\n"],
        },
        {
            "cell_type": "code",
            "execution_count": 12,
            "metadata": {},
            "outputs": [],
            "source": ["plt.show()\n", "print('finished')\n"],
        },
    ]
    path.write_text(json.dumps(_notebook_doc(cells)), encoding="utf-8")


def test_module_import_avoids_nbconvert() -> None:
    # Lazy import: importing the module must not pull in the heavy nbconvert.
    assert "nbconvert" not in sys.modules


def test_convert_notebook_strips_interactive_code(tmp_path: Path) -> None:
    notebook = tmp_path / "test.ipynb"
    output = tmp_path / "test.py"
    _write_demo_notebook(notebook)

    assert convert_notebook(notebook, output) == output
    code = output.read_text()

    for keep in ("import torch", "import matplotlib.pyplot as plt", "x = torch.randn(10)", "print('finished')"):
        assert keep in code
    for strip in ("IPython", "ipywidgets", "clear_output", "plt.show()"):
        assert strip not in code


def test_convert_notebook_preserves_previous_output(tmp_path: Path) -> None:
    notebook = tmp_path / "test.ipynb"
    output = tmp_path / "test.py"
    _write_demo_notebook(notebook)
    output.write_text("old content", encoding="utf-8")

    convert_notebook(notebook, output)

    backup = output.with_name(output.name + ".old")
    assert backup.exists()
    assert backup.read_text() == "old content"
