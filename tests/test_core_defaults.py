"""Tests for resource default derivation."""

import json
from pathlib import Path

import pytest

from lazy107.core.defaults import (
    GPU_PACKAGES,
    _extract_package_names,
    derive_defaults,
    detect_ddp,
    detect_dependencies,
    has_gpu_dependency,
    scan_imports,
    write_requirements,
)


def test_extract_package_names_strips_comments_and_version_specs() -> None:
    text = """
# optional extras
rich>=13.0
requests==2.31
pandas
nbformat  # notebook plumbing
"""
    assert _extract_package_names(text) == {"rich", "requests", "pandas", "nbformat"}


def test_detect_dependencies_reads_pyproject_deps(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sandbox"\ndependencies = ["torch", "numpy>=1.24"]\n',
        encoding="utf-8",
    )
    deps = detect_dependencies(tmp_path)
    assert "torch" in deps
    assert "numpy" in deps


def test_detect_dependencies_reads_requirements_files(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("scipy==1.11\nmatplotlib\n", encoding="utf-8")
    deps = detect_dependencies(tmp_path)
    assert "scipy" in deps
    assert "matplotlib" in deps


def test_detect_dependencies_combines_all_sources(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sandbox"\ndependencies = ["torch"]\n',
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    deps = detect_dependencies(tmp_path)
    assert {"torch", "numpy"} <= deps


def test_detect_dependencies_from_environment_yml(tmp_path: Path) -> None:
    (tmp_path / "environment.yml").write_text(
        "name: sandbox\ndependencies:\n  - python=3.12\n  - torch\n  - numpy\n",
        encoding="utf-8",
    )
    deps = detect_dependencies(tmp_path)
    assert "torch" in deps
    assert "numpy" in deps


def test_gpu_packages_includes_major_frameworks() -> None:
    assert {"torch", "tensorflow", "jax"} <= GPU_PACKAGES


@pytest.mark.parametrize(
    ("packages", "expected"),
    [
        ({"torch", "numpy"}, True),
        ({"tensorflow"}, True),
        ({"jaxlib"}, True),
        ({"numpy", "scikit-learn"}, False),
        (set(), False),
    ],
)
def test_has_gpu_dependency_table(packages: set[str], expected: bool) -> None:
    assert has_gpu_dependency(packages) is expected


def test_derive_defaults_prefers_gpu_profile(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sandbox"\ndependencies = ["torch", "torchvision"]\n',
        encoding="utf-8",
    )
    defaults = derive_defaults(tmp_path)
    assert defaults == {"gpu": 1, "cpus": 4, "mem": "16G", "time": "2:00:00"}


def test_derive_defaults_falls_back_to_cpu_profile(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("scikit-learn\nnumpy\n", encoding="utf-8")
    defaults = derive_defaults(tmp_path)
    assert defaults == {"gpu": 0, "cpus": 2, "mem": "4G", "time": "1:00:00"}


def test_derive_defaults_without_dependency_files(tmp_path: Path) -> None:
    assert derive_defaults(tmp_path)["gpu"] == 0


def test_scan_imports_maps_mismatches_and_identity_fallback(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text(
        "import torch\nimport sklearn\nimport requests\nimport os\nfrom . import util\n",
        encoding="utf-8",
    )
    assert scan_imports(tmp_path) == {"torch", "scikit-learn", "requests"}


def test_scan_imports_skips_local_modules(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text("import torch\n", encoding="utf-8")
    (tmp_path / "train.py").write_text("import model\nfrom model import build\n", encoding="utf-8")
    assert scan_imports(tmp_path) == {"torch"}


def test_scan_imports_handles_utf8_bom(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_bytes(b"\xef\xbb\xbfimport torch\n")
    assert scan_imports(tmp_path) == {"torch"}


def test_scan_imports_skips_hidden_dirs(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("import pandas\n", encoding="utf-8")
    assert scan_imports(tmp_path) == set()


def test_scan_imports_tolerates_unreadable_dirs(tmp_path: Path, blocked_dir: Path) -> None:
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    assert scan_imports(tmp_path) == {"torch"}


def test_scan_imports_captures_comma_separated_imports(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("import numpy, pandas\n", encoding="utf-8")
    assert scan_imports(tmp_path) == {"numpy", "pandas"}


def test_scan_imports_tolerates_malformed_import_lines(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text(
        "import torch,\nimport ,\nimport \nfrom \nimport numpy\n", encoding="utf-8"
    )
    assert scan_imports(tmp_path) == {"torch", "numpy"}


def test_scan_imports_ignores_undecodable_files(tmp_path: Path) -> None:
    (tmp_path / "legacy.py").write_bytes("import torch\n# \u4e2d\u6587\u6ce8\u91ca\n".encode("gbk"))
    (tmp_path / "train.py").write_text("import numpy\n", encoding="utf-8")
    assert scan_imports(tmp_path) == {"numpy"}


def test_scan_imports_reads_notebook_code_cells(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["import pandas  # text only"]},
            {"cell_type": "code", "source": ["import torch\n", "from sklearn import svm\n"]},
        ]
    }
    (tmp_path / "train.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    assert scan_imports(tmp_path) == {"torch", "scikit-learn"}


def test_detect_dependencies_includes_scanned_imports(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    assert "torch" in detect_dependencies(tmp_path)


def test_derive_defaults_for_scanned_gpu_project(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    assert derive_defaults(tmp_path) == {"gpu": 1, "cpus": 4, "mem": "16G", "time": "2:00:00"}


def test_write_requirements(tmp_path: Path) -> None:
    path = write_requirements(tmp_path, {"numpy", "torch"})
    assert path == tmp_path / "requirements.txt"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# generated by lazy107 env")
    assert "numpy" in text
    assert "torch" in text


def test_detect_ddp_torch_distributed_import(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("import torch.distributed as dist\n", encoding="utf-8")
    assert detect_ddp(tmp_path) is True


def test_detect_ddp_ddp_class_usage(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text(
        "from torch.nn.parallel import DistributedDataParallel\n"
        "model = DistributedDataParallel(model)\n",
        encoding="utf-8",
    )
    assert detect_ddp(tmp_path) is True


def test_detect_ddp_init_process_group(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("dist.init_process_group(\"nccl\")\n", encoding="utf-8")
    assert detect_ddp(tmp_path) is True


def test_detect_ddp_accelerate_and_deepspeed(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("from accelerate import Accelerator\nimport deepspeed\n", encoding="utf-8")
    assert detect_ddp(tmp_path) is True


def test_detect_ddp_lightning_ddp_strategy(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text('trainer = pl.Trainer(strategy="ddp")\n', encoding="utf-8")
    assert detect_ddp(tmp_path) is True


def test_detect_ddp_plain_torch_false(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("import torch\nimport torch.nn as nn\n", encoding="utf-8")
    assert detect_ddp(tmp_path) is False


def test_detect_ddp_ignores_comments_and_dataparallel(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text(
        "# run with torchrun if you like\n"
        "# model = DistributedDataParallel(model)\n"
        "# dist.init_process_group('nccl')\n"
        "from torch.nn import DataParallel\n"
        "model = DataParallel(model)  # single-process multi-GPU\n",
        encoding="utf-8",
    )
    assert detect_ddp(tmp_path) is False


def test_detect_ddp_skips_hidden_dirs(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("import torch.distributed\n", encoding="utf-8")
    assert detect_ddp(tmp_path) is False


def test_detect_ddp_reads_notebook_cells(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["DistributedDataParallel is great"]},
            {"cell_type": "code", "source": ["import torch.distributed as dist\n"]},
        ]
    }
    (tmp_path / "train.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    assert detect_ddp(tmp_path) is True
