"""Tests for project scaffolding (package-data templates)."""

from pathlib import Path

import pytest

from lazy107.template import scaffold, template_dir


def test_template_dir_is_inside_package() -> None:
    assert "lazy107" in str(template_dir())
    assert (template_dir() / "environment.yml").exists()


def test_scaffold_creates_expected_files(tmp_path: Path) -> None:
    target = tmp_path / "new_project"
    scaffold(target, "new_project")

    expected_files = [
        ".gitignore",
        "README.md",
        "environment.yml",
        "pyproject.toml",
        "scripts/train.sbatch",
        "src/__init__.py",
        "src/data.py",
        "src/model.py",
        "src/train.py",
    ]
    for rel_path in expected_files:
        assert (target / rel_path).exists(), f"Missing file: {rel_path}"

    data_py = (target / "src" / "data.py").read_text()
    assert "write dataset loading code in src/data.py" in data_py

    train_py = (target / "src" / "train.py").read_text()
    for import_line in ("from src.data import load_data", "from src.model import build_model"):
        assert import_line in train_py


def test_scaffold_substitutes_project_name(tmp_path: Path) -> None:
    target = tmp_path / "my-project"
    scaffold(target, "my-project")
    pyproject = (target / "pyproject.toml").read_text()
    assert 'name = "my-project"' in pyproject
    assert "{project}" not in pyproject
    env_yml = (target / "environment.yml").read_text()
    assert "name: my-project" in env_yml
    sbatch = (target / "scripts" / "train.sbatch").read_text()
    assert "conda activate my-project" in sbatch


def test_scaffold_raises_on_nonempty_dir(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    (target / "placeholder.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="non-empty"):
        scaffold(target, "existing")


def test_scaffold_allows_empty_dir(tmp_path: Path) -> None:
    target = tmp_path / "empty"
    target.mkdir()
    scaffold(target, "empty")
    assert (target / "src" / "train.py").exists()


def test_scaffold_skips_cache_junk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lint run inside the package must never poison scaffolds (binary cache files)."""
    fake_template = tmp_path / "template"
    (fake_template / "src").mkdir(parents=True)
    (fake_template / "src" / "train.py").write_text("print('hi')")
    (fake_template / ".ruff_cache").mkdir()
    (fake_template / ".ruff_cache" / "cache.bin").write_bytes(b"\xff\x00\xfe")
    (fake_template / "__pycache__").mkdir()
    (fake_template / "__pycache__" / "m.pyc").write_bytes(b"\x00\x01")
    monkeypatch.setattr("lazy107.template.template_dir", lambda: fake_template)

    target = tmp_path / "proj"
    scaffold(target, "proj")

    assert (target / "src" / "train.py").exists()
    assert not (target / ".ruff_cache").exists()
    assert not (target / "__pycache__").exists()
