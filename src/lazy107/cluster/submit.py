"""Cluster-side I/O: sbatch submission and Slurm preflight checks."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class SubmitError(RuntimeError):
    """Raised when sbatch fails or its output cannot be parsed."""


def submit_sbatch(project_root: Path, sbatch_path: Path) -> str:
    """Submit a batch script and return the job id from sbatch's reply."""
    result = subprocess.run(
        ["sbatch", str(sbatch_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SubmitError(f"sbatch rejected the job (exit {result.returncode}): {detail}")
    fields = result.stdout.strip().split()
    if len(fields) < 4 or fields[:3] != ["Submitted", "batch", "job"]:
        raise SubmitError(f"unexpected sbatch response: {result.stdout.strip()}")
    return fields[3]


def slurm_check(partition: str) -> bool:
    """True when `sinfo -p <partition> -h` reports a usable partition."""
    try:
        probe = subprocess.run(
            ["sinfo", "-p", partition, "-h"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("sinfo is not installed; skipping the partition check.")
        return False
    if probe.returncode == 0 and probe.stdout.strip():
        return True
    logger.warning(
        "partition check failed for '%s' (sinfo exit code %d)",
        partition,
        probe.returncode,
    )
    if probe.stderr.strip():
        logger.warning("sinfo stderr: %s", probe.stderr.strip())
    return False
