"""Tests for submission and Slurm preflight."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lazy107.cluster.submit import SubmitError, slurm_check, submit_sbatch


def test_submit_sbatch_parses_job_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="Submitted batch job 12345\n", stderr="", returncode=0)

    monkeypatch.setattr("lazy107.cluster.submit.subprocess.run", fake_run)
    assert submit_sbatch(tmp_path, tmp_path / "train.sbatch") == "12345"


def test_submit_sbatch_bad_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="some error\n", stderr="", returncode=0)

    monkeypatch.setattr("lazy107.cluster.submit.subprocess.run", fake_run)
    with pytest.raises(SubmitError, match="unexpected sbatch response"):
        submit_sbatch(tmp_path, tmp_path / "train.sbatch")


def test_submit_sbatch_nonzero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="", stderr="sbatch: error: Invalid partition", returncode=1)

    monkeypatch.setattr("lazy107.cluster.submit.subprocess.run", fake_run)
    with pytest.raises(SubmitError, match="sbatch rejected .*Invalid partition"):
        submit_sbatch(tmp_path, tmp_path / "train.sbatch")


def test_slurm_check_valid() -> None:
    mock_result = SimpleNamespace(
        returncode=0,
        stdout="Students*     up    7:00:00      1/0/1/2  gpu:2080ti  (null)\n",
        stderr="",
    )
    with patch("lazy107.cluster.submit.subprocess.run", return_value=mock_result) as mock_run:
        assert slurm_check("Students") is True
        mock_run.assert_called_once_with(
            ["sinfo", "-p", "Students", "-h"],
            capture_output=True,
            text=True,
            check=False,
        )


def test_slurm_check_invalid(caplog: pytest.LogCaptureFixture) -> None:
    mock_result = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="sinfo: error: Invalid partition specified: NonExistent\n",
    )
    with patch("lazy107.cluster.submit.subprocess.run", return_value=mock_result):
        assert slurm_check("NonExistent") is False

    assert "partition check failed" in caplog.text
    assert "NonExistent" in caplog.text
    assert "sinfo exit code 1" in caplog.text


def test_slurm_check_sinfo_missing(caplog: pytest.LogCaptureFixture) -> None:
    with patch(
        "lazy107.cluster.submit.subprocess.run",
        side_effect=FileNotFoundError("sinfo not found"),
    ):
        assert slurm_check("Students") is False
    assert "sinfo is not installed" in caplog.text
