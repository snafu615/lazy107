"""Tests for the runs ledger."""

from pathlib import Path

from lazy107.core.plan import RunPlan
from lazy107.core.record import (
    find_array,
    read_history,
    record_command,
    record_run,
    recorded_runs,
)


def test_record_run_appends(tmp_path: Path) -> None:
    notes = tmp_path / "notes" / "runs.md"
    plan = RunPlan(entry="train.py", partition="Students", qos="qos_stu_default")

    record_run(notes_file=notes, plan=plan, job_id="54321", sbatch_path=tmp_path / "scripts" / "train.sbatch")

    content = notes.read_text()
    assert "## Job 54321" in content
    assert "54321" in content
    assert "train.py" in content
    assert "Students" in content
    assert "qos_stu_default" in content


def test_record_run_notes_array(tmp_path: Path) -> None:
    notes = tmp_path / "notes" / "runs.md"
    plan = RunPlan(entry="train.py", array="1-5%2")
    record_run(notes_file=notes, plan=plan, job_id="54321", sbatch_path=tmp_path / "scripts" / "train.sbatch")
    assert "**Task array**: 1-5%2" in notes.read_text()


def test_find_array_returns_recorded_spec(tmp_path: Path) -> None:
    notes = tmp_path / "notes" / "runs.md"
    record_run(
        notes_file=notes,
        plan=RunPlan(entry="train.py", array="1-5%2"),
        job_id="54321",
        sbatch_path=tmp_path / "scripts" / "train.sbatch",
    )
    assert find_array(notes, "54321") == "1-5%2"


def test_find_array_empty_when_unknown_or_missing(tmp_path: Path) -> None:
    notes = tmp_path / "notes" / "runs.md"
    record_run(
        notes_file=notes,
        plan=RunPlan(entry="train.py", array="1-5%2"),
        job_id="54321",
        sbatch_path=tmp_path / "scripts" / "train.sbatch",
    )
    assert find_array(notes, "99999") == ""
    assert find_array(tmp_path / "missing" / "runs.md", "54321") == ""


def test_recorded_runs_lists_ids_in_order(tmp_path: Path) -> None:
    notes = tmp_path / "notes" / "runs.md"
    for job_id in ("41", "42"):
        record_run(notes_file=notes, plan=RunPlan(entry="train.py"), job_id=job_id, sbatch_path=tmp_path / "x.sbatch")
    assert recorded_runs(notes) == ["41", "42"]


def test_recorded_runs_empty_when_missing(tmp_path: Path) -> None:
    assert recorded_runs(tmp_path / "notes" / "runs.md") == []


def test_record_command_appends_with_exit_code(tmp_path: Path) -> None:
    history = tmp_path / "notes" / "history.md"
    record_command(history, ["plan", "--entry", "train.py"], 0)
    record_command(history, ["env", "--yes"], 1)
    lines = read_history(history)
    assert len(lines) == 2
    assert "lazy107 plan --entry train.py (exit 0)" in lines[0]
    assert "lazy107 env --yes (exit 1)" in lines[1]


def test_read_history_empty_when_missing(tmp_path: Path) -> None:
    assert read_history(tmp_path / "notes" / "history.md") == []


def test_read_history_ignores_non_entries(tmp_path: Path) -> None:
    history = tmp_path / "notes" / "history.md"
    history.parent.mkdir(parents=True)
    history.write_text("# History\n\n", encoding="utf-8")
    record_command(history, ["check"], 0)
    assert len(read_history(history)) == 1


def test_recorded_runs_reads_pre_scrub_format(tmp_path: Path) -> None:
    notes = tmp_path / "notes" / "runs.md"
    notes.parent.mkdir(parents=True)
    notes.write_text(
        "## Run 41\n\n"
        "- **Time**: 2026-09-01 00:00:00 UTC\n"
        "- **Job ID**: 41\n"
        "- **Entry**: train.py\n"
        "- **Array**: 1-5%2\n\n"
        "## Job 42\n\n"
        "- **Submitted at**: 2026-09-02T00:00:00Z\n"
        "- **Entry point**: train.py\n\n",
        encoding="utf-8",
    )
    assert recorded_runs(notes) == ["41", "42"]
    assert find_array(notes, "41") == "1-5%2"
    assert find_array(notes, "42") == ""
