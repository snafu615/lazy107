"""Runs ledger: append markdown entries to notes/runs.md."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lazy107.core.plan import RunPlan


def record_run(notes_file: Path, plan: RunPlan, job_id: str, sbatch_path: Path) -> None:
    notes_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    account_line = f"- **Account**: {plan.account}\n" if plan.account else ""
    array_line = f"- **Task array**: {plan.array}\n" if plan.array else ""
    entry = (
        f"## Job {job_id}\n\n"
        f"- **Submitted at**: {timestamp}\n"
        f"- **Batch script**: {sbatch_path}\n"
        f"- **Entry point**: {plan.entry}\n"
        f"{account_line}{array_line}- **Partition**: {plan.partition}\n"
        f"- **QoS**: {plan.qos}\n\n"
    )
    with notes_file.open("a", encoding="utf-8") as f:
        f.write(entry)


def recorded_runs(notes_file: Path) -> list[str]:
    """Job ids recorded in the runs ledger, in submission order.

    Accepts both the current "## Job" and the pre-scrub "## Run" format
    so ledgers written by older versions stay visible.
    """
    if not notes_file.exists():
        return []
    return [
        line[7:].strip()
        for line in notes_file.read_text(encoding="utf-8").splitlines()
        if line.startswith(("## Job ", "## Run "))
    ]


def find_array(notes_file: Path, job_id: str) -> str:
    """Array spec recorded for a job id in the runs ledger; '' when none.

    `submit --array` one-offs are recoverable here, so `watch`/`logs`
    target the per-task log set even when 107.toml sets no `array`.
    """
    if not notes_file.exists():
        return ""
    capture = False
    for line in notes_file.read_text(encoding="utf-8").splitlines():
        if line.startswith(("## Job ", "## Run ")):
            capture = line.strip() in (f"## Job {job_id}", f"## Run {job_id}")
        elif capture:
            if line.startswith("## "):
                break
            if line.startswith(("- **Task array**: ", "- **Array**: ")):
                return line.split(": ", 1)[1].strip()
    return ""


def record_command(history_file: Path, argv: list[str], rc: int) -> None:
    """Append one CLI invocation (with its exit code) to notes/history.md."""
    history_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with history_file.open("a", encoding="utf-8") as f:
        f.write(f"- [{timestamp}] lazy107 {' '.join(argv)} (exit {rc})\n")


def read_history(history_file: Path) -> list[str]:
    """Recorded command lines, oldest first."""
    if not history_file.exists():
        return []
    return [
        line.removeprefix("- ").strip()
        for line in history_file.read_text(encoding="utf-8").splitlines()
        if line.startswith("- ")
    ]
