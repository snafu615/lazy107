"""Job monitoring: command generation and sacct log lookup."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def log_globs(log_dir: str, job_name: str, job_id: str, array: str = "") -> tuple[str, str]:
    """(stdout, stderr) log glob patterns, relative to the project root.

    Python tracebacks and slurmstepd errors land in the .err file, so both
    halves matter when diagnosing a failed job.
    """
    base = f"{job_name}_{job_id}"
    wild = "_*" if array else ""
    return f"{log_dir}/{base}{wild}.out", f"{log_dir}/{base}{wild}.err"


def monitor_commands(job_id: str, job_name: str, log_dir: str, array: str = "") -> list[str]:
    # Array jobs log per task (%A_%a), so the tails target the whole set.
    out, err = log_globs(log_dir, job_name, job_id, array)
    return [
        f"scontrol show job {job_id}",
        "squeue --me",
        f"tail -f {out}",
        f"tail -f {err}",
    ]


def sacct_row(job_id: str) -> tuple[str, str] | None:
    """(State, ExitCode) for the job allocation from sacct, or None when unavailable."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "-X", "-n", "-o", "State,ExitCode", "--parsable2"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("sacct not found; skipping exit-code lookup")
        return None
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2 and parts[0]:
            return parts[0], parts[1]
    return None


def collect_logs(
    project_root: Path,
    log_dir: str,
    job_name: str,
    job_id: str,
    array: str = "",
    max_bytes: int = 65536,
) -> str:
    """Tail of the job's .out/.err logs (newest file last), capped at max_bytes.

    Both halves matter: normal output goes to .out while tracebacks and
    slurmstepd errors land in .err. Missing files are skipped silently.
    """
    chunks: list[str] = []
    for pattern in log_globs(log_dir, job_name, job_id, array):
        for path in sorted(project_root.glob(pattern)):
            try:
                chunk = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if chunk.strip():
                chunks.append(chunk)
    text = "\n".join(chunks)
    if len(text) > max_bytes:
        text = "... (truncated) ...\n" + text[-max_bytes:]
    return text
