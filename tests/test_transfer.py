"""Tests for the transfer checklist (large-file discovery, print-only)."""

from pathlib import Path

from lazy107 import template as template_mod
from lazy107.transfer import LARGE_DIRS, LARGE_SUFFIXES, checklist, large_files


def test_large_files_empty(tmp_path: Path) -> None:
    assert large_files(tmp_path) == []
    assert checklist(tmp_path) == ["No large files found; nothing to transfer."]


def test_large_files_tolerates_unreadable_dirs(tmp_path: Path, blocked_dir: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw.bin").write_bytes(b"x")
    assert large_files(tmp_path) == [Path("data/raw.bin")]


def test_large_files_discovers_suffixes_and_dirs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text("x", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw.bin").write_bytes(b"x")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "net.pt").write_bytes(b"x")
    (tmp_path / "weights.pth").write_bytes(b"x")

    files = large_files(tmp_path)
    assert files == [
        Path("data/raw.bin"),
        Path("models/net.pt"),
        Path("weights.pth"),
    ]


def test_checklist_tar_and_sha256(tmp_path: Path) -> None:
    (tmp_path / "weights.pth").write_bytes(b"x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.npy").write_bytes(b"x")

    lines = checklist(tmp_path)
    text = "\n".join(lines)
    assert f"{tmp_path.name}-data.tar.gz" in text
    assert 'tar -czf' in text
    assert 'weights.pth' in text
    assert 'sha256sum' in text


def test_checklist_excludes_own_archive(tmp_path: Path) -> None:
    (tmp_path / "weights.pth").write_bytes(b"x")
    (tmp_path / f"{tmp_path.name}-data.tar.gz").write_bytes(b"x")
    files = large_files(tmp_path)
    assert all(not f.name.endswith("-data.tar.gz") for f in files)


def test_transfer_patterns_match_template_gitignore() -> None:
    """Fix #10: one pattern set; template .gitignore must cover every transfer pattern."""
    gitignore = (template_mod.template_dir() / ".gitignore").read_text(encoding="utf-8")
    for suffix in LARGE_SUFFIXES:
        assert f"*{suffix}" in gitignore, f"gitignore missing {suffix}"
    for d in LARGE_DIRS:
        assert f"{d}/" in gitignore, f"gitignore missing {d}/"
