"""Job failure diagnosis: sacct exit code + log text -> cause + fix.

Two evidence layers (see examples/failures/SIGNATURES.md, the live harvest
on the 107 cluster, 2026-09-03):
- standardized Slurm: sacct ExitCode semantics (0:9 SIGKILL, 0:15 SIGTERM);
- log signatures: failure text in logs/<job>.err — tracebacks, the
  timestamp-prefixed "DUE TO TIME LIMIT" line, torch's CUDA OOM message.

Pure logic: no I/O, no subprocess — fully unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnosis:
    cause: str
    evidence: str
    fix: str


# Standardized Slurm exit codes ("code:signal" as reported by sacct).
_EXIT_CODES: dict[str, tuple[str, str]] = {
    "1:0": (
        "the job's own code exited non-zero",
        "read the traceback in the .err log",
    ),
    "0:9": (
        "killed by SIGKILL (usually out of memory)",
        "raise `mem` in 107.toml or reduce memory use",
    ),
    "0:15": (
        "terminated by SIGTERM (wall-time limit or manual cancel)",
        "raise `time` in 107.toml, add checkpointing, or reduce the workload",
    ),
}


def exit_code_diagnosis(exit_code: str | None) -> Diagnosis | None:
    if exit_code is None:
        return None
    cause, fix = _EXIT_CODES.get(exit_code, (None, None))
    if cause is None:
        return None
    return Diagnosis(cause, f"sacct ExitCode={exit_code}", fix)


# (pattern, cause, fix) — cause/fix format strings may use {0}, {1}, ... for
# regex groups. Ordered most-specific first; the generic traceback matcher is
# last. Prefix-agnostic: lines may carry a plain `slurmstepd:` prefix or the
# timestamped `[2026-09-03T...] error:` format seen on 107.
_MATCHERS: list[tuple[str, str, str]] = [
    (
        r"ModuleNotFoundError: No module named '(\w+)'",
        "missing dependency '{0}'",
        "add {0} to requirements.txt/environment.yml and re-run `lazy107 env --yes`",
    ),
    (
        r"FileNotFoundError: \[Errno 2\] No such file or directory: '(.+)'",
        "missing data file '{0}'",
        "upload it to the cluster (see `lazy107 transfer`)",
    ),
    (
        r"ImportError: cannot import name '(\w+)' from '(\S+)'",
        "cannot import '{0}' from '{1}'",
        "check the import name in the entry file",
    ),
    (
        r"\*\*\* JOB \d+ ON \S+ CANCELLED AT \S+ DUE TO TIME LIMIT \*\*\*",
        "hit the wall-time limit",
        "raise `time` in 107.toml, add checkpointing, or reduce the workload",
    ),
    (
        r"Detected \d+ oom-kill event\(s\)",
        "OOM-killed by Slurm (memory limit)",
        "raise `mem` in 107.toml or reduce memory use",
    ),
    (
        (
            r"torch\.OutOfMemoryError: CUDA out of memory\. Tried to allocate ([\d.]+ \S+)\. "
            r"GPU \d+ has a total capacity of ([\d.]+ GiB) of which ([\d.]+ GiB) is free"
        ),
        "CUDA out of memory: tried to allocate {0} but only {2} free (GPU total {1})",
        "reduce batch size / model size, or use gradient accumulation",
    ),
    (
        r"(?m)^(\w[\w.]*(?:Error|Exception)): (.*)$",
        "{0}: {1}",
        "check the traceback above",
    ),
]


def diagnose(log_text: str, exit_code: str | None = None) -> Diagnosis | None:
    """First matching log signature wins; falls back to sacct exit-code semantics."""
    for pattern, cause, fix in _MATCHERS:
        match = re.search(pattern, log_text)
        if match:
            return Diagnosis(cause.format(*match.groups()), match.group(0), fix.format(*match.groups()))
    return exit_code_diagnosis(exit_code)
