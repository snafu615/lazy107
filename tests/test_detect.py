"""Tests for entry-point detection."""

from pathlib import Path

from lazy107.core.detect import detect_entries, recommend_entry


def test_detect_entries_py_and_ipynb(tmp_path: Path) -> None:
    (tmp_path / "main.py").touch()
    (tmp_path / "train.py").touch()
    (tmp_path / "z.ipynb").touch()
    (tmp_path / "a.ipynb").touch()

    assert detect_entries(tmp_path) == [
        tmp_path / "train.py",
        tmp_path / "main.py",
        tmp_path / "a.ipynb",
        tmp_path / "z.ipynb",
    ]


def test_detect_entries_nested(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").touch()
    (tmp_path / "utils.py").touch()

    result = detect_entries(tmp_path)
    assert result[0] == tmp_path / "src" / "train.py"
    assert tmp_path / "utils.py" in result


def test_detect_entries_empty(tmp_path: Path) -> None:
    assert detect_entries(tmp_path) == []


def test_detect_entries_tolerates_unreadable_dirs(tmp_path: Path, blocked_dir: Path) -> None:
    (tmp_path / "train.py").touch()
    assert detect_entries(tmp_path) == [tmp_path / "train.py"]


def test_recommend_entry_prefers_train(tmp_path: Path) -> None:
    (tmp_path / "train.py").touch()
    (tmp_path / "main.py").touch()
    assert recommend_entry(tmp_path) == tmp_path / "train.py"


def test_recommend_entry_fallback_to_main(tmp_path: Path) -> None:
    (tmp_path / "main.py").touch()
    assert recommend_entry(tmp_path) == tmp_path / "main.py"


def test_recommend_entry_notebook(tmp_path: Path) -> None:
    (tmp_path / "analysis.ipynb").touch()
    assert recommend_entry(tmp_path) == tmp_path / "analysis.ipynb"


def test_recommend_entry_none(tmp_path: Path) -> None:
    assert recommend_entry(tmp_path) is None
