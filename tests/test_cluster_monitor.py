"""Tests for monitoring command generators, log collection, and sacct lookup."""

from pathlib import Path
from types import SimpleNamespace

from lazy107.cluster.monitor import collect_logs, log_globs, monitor_commands, sacct_row


def test_monitor_commands_returns_expected_pipeline() -> None:
    cmds = monitor_commands(job_id="77777", job_name="train", log_dir="logs")
    assert cmds == [
        "scontrol show job 77777",
        "squeue --me",
        "tail -f logs/train_77777.out",
        "tail -f logs/train_77777.err",
    ]


def test_monitor_commands_array_job_tails_all_tasks() -> None:
    cmds = monitor_commands(job_id="12345", job_name="train", log_dir="logs", array="1-5")
    assert cmds[-2:] == ["tail -f logs/train_12345_*.out", "tail -f logs/train_12345_*.err"]


def test_log_globs_pairs_out_and_err() -> None:
    assert log_globs("logs", "train", "12345") == ("logs/train_12345.out", "logs/train_12345.err")
    assert log_globs("logs", "train", "12345", array="1-5") == (
        "logs/train_12345_*.out",
        "logs/train_12345_*.err",
    )


def test_collect_logs_reads_both_files(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "train_12345.out").write_text("epoch 1\n", encoding="utf-8")
    (logs / "train_12345.err").write_text("ModuleNotFoundError: No module named 'x'\n", encoding="utf-8")
    text = collect_logs(tmp_path, "logs", "train", "12345")
    assert "epoch 1" in text
    assert "ModuleNotFoundError" in text


def test_collect_logs_skips_missing_files(tmp_path: Path) -> None:
    assert collect_logs(tmp_path, "logs", "train", "12345") == ""


def test_sacct_row_parses_state_and_exit(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="FAILED|1:0\n", stderr="", returncode=0)

    monkeypatch.setattr("lazy107.cluster.monitor.subprocess.run", fake_run)
    assert sacct_row("12345") == ("FAILED", "1:0")


def test_sacct_row_none_when_sacct_missing(monkeypatch) -> None:
    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("lazy107.cluster.monitor.subprocess.run", raise_fnf)
    assert sacct_row("12345") is None
