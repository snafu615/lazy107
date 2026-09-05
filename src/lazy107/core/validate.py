"""Contract checks and injection safety for RunPlan values."""

from __future__ import annotations

import re

from lazy107.core.plan import RunPlan

_SHELL_METACHARS = re.compile(r"[\s;&|<>$`(){}*?!\[\]#~'\"\\]")

_REQUIRED_SAFE = ("entry", "partition", "qos", "account", "mem", "time", "log_dir", "conda_env", "job_name")

# Slurm --array syntax: N, N-M, N-M:S, comma lists, optional %K max-concurrent cap.
# Step and %K cap must be >= 1 (sbatch rejects :0 and %0).
_ARRAY_RE = re.compile(
    r"^[0-9]+(?:-[0-9]+(?::[1-9][0-9]*)?)?(?:,[0-9]+(?:-[0-9]+(?::[1-9][0-9]*)?)?)*(?:%[1-9][0-9]*)?$"
)


def is_safe(value: str) -> bool:
    """True for non-empty values free of whitespace and shell metacharacters."""
    return bool(value) and not _SHELL_METACHARS.search(value)


def gpu_preflight_error(gpu: int, torch_cuda: str | None) -> str | None:
    """Error message when gpu>0 but the env's torch build is CPU-only."""
    if gpu > 0 and torch_cuda == "":
        return (
            "GPU requested but the env's torch is a CPU-only build; "
            "reinstall via `lazy107 env --yes` (PyPI default wheel bundles CUDA)"
        )
    return None


def validate_plan(plan: RunPlan, ddp: bool = False) -> list[str]:
    errors = []
    for field in _REQUIRED_SAFE:
        if getattr(plan, field) and not is_safe(getattr(plan, field)):
            errors.append(f"{field}: must be free of whitespace and shell metacharacters")
    if not plan.entry:
        errors.append("entry: must not be empty")
    if plan.gpu < 0:
        errors.append("gpu: must be >= 0")
    if plan.cpus < 1:
        errors.append("cpus: must be >= 1")
    if plan.nodes < 1:
        errors.append("nodes: must be >= 1")
    if plan.ntasks < 1:
        errors.append("ntasks: must be >= 1")
    if plan.array and not _ARRAY_RE.match(plan.array):
        errors.append("array: must look like 1-5%2 or 0,2,4 (Slurm --array syntax)")
    if ddp and plan.gpu > 0 and plan.nodes > 1 and plan.ntasks != plan.nodes:
        errors.append(
            f"ntasks ({plan.ntasks}) must equal nodes ({plan.nodes}) for multi-node DDP "
            "(torchrun needs one launcher per node)"
        )
    return errors
