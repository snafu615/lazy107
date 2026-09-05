"""RunPlan -> sbatch script rendering (deterministic string function)."""

from __future__ import annotations

from pathlib import Path

from lazy107.core.plan import RunPlan

# Only runs when torch is importable in the job env; fails fast instead of
# silently training on CPU (jax/tf projects skip the torch-specific assert).
_GPU_GUARD = (
    "python - <<'PY'\n"
    "import importlib.util\n"
    "if importlib.util.find_spec('torch'):\n"
    "    import torch\n"
    "    assert torch.cuda.is_available(), 'GPU requested but CUDA unavailable in this job'\n"
    "PY\n"
    "\n"
)


def _conda_block(env_name: str) -> str:
    if not env_name:
        return ""
    # conda scripts touch unset vars; guard against `set -u` before sourcing.
    return (
        "set +u\n"
        'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
        f"conda activate {env_name}\n"
        "set -u\n"
        "\n"
    )


def _launch_line(plan: RunPlan, ddp: bool) -> str:
    """The command that runs the entry.

    An explicit `command` always wins (full control, used verbatim).
    Otherwise a DDP GPU job launches under torchrun so that `gpu=N` actually
    runs N processes: `--standalone` on a single node; on multiple nodes
    `srun` starts one torchrun per node with a c10d rendezvous (the
    PyTorch-documented Slurm recipe).
    """
    if plan.command:
        return plan.command
    if ddp and plan.gpu > 0:
        if plan.nodes == 1:
            return f"torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE {plan.entry}"
        return (
            "srun torchrun --nnodes=$SLURM_JOB_NUM_NODES "
            "--nproc_per_node=$SLURM_GPUS_ON_NODE "
            "--rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d "
            f'--rdzv_endpoint="$(scontrol show hostname $SLURM_NODELIST | head -n1):29500" {plan.entry}'
        )
    return f"python {plan.entry}"


def render_sbatch(plan: RunPlan, ddp: bool = False) -> str:
    gpu_line = f"#SBATCH --gres=gpu:{plan.gpu}\n" if plan.gpu > 0 else ""
    account_line = f"#SBATCH --account={plan.account}\n" if plan.account else ""
    guard = _GPU_GUARD if plan.gpu > 0 else ""
    # Multi-node DDP needs exactly one torchrun launcher per node.
    ntasks_per_node = "#SBATCH --ntasks-per-node=1\n" if ddp and plan.gpu > 0 and plan.nodes > 1 else ""
    # Array jobs get per-task logs (%A = master job id, %a = task index) so
    # concurrent tasks never interleave in one file.
    array_line = f"#SBATCH --array={plan.array}\n" if plan.array else ""
    log_id = "%A_%a" if plan.array else "%j"
    array_hint = "# array task index available as $SLURM_ARRAY_TASK_ID\n" if plan.array else ""
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={plan.effective_job_name}\n"
        f"{account_line}#SBATCH --partition={plan.partition}\n"
        f"#SBATCH --qos={plan.qos}\n"
        f"#SBATCH --nodes={plan.nodes}\n"
        f"#SBATCH --ntasks={plan.ntasks}\n"
        f"{ntasks_per_node}{array_line}#SBATCH --cpus-per-task={plan.cpus}\n"
        f"#SBATCH --mem={plan.mem}\n"
        f"{gpu_line}#SBATCH --time={plan.time}\n"
        f"#SBATCH --output={plan.log_dir}/%x_{log_id}.out\n"
        f"#SBATCH --error={plan.log_dir}/%x_{log_id}.err\n"
        "\n"
        "set -euo pipefail\n"
        'cd "$SLURM_SUBMIT_DIR"\n'
        f"mkdir -p {plan.log_dir}\n"
        "\n"
        f"{_conda_block(plan.conda_env)}{guard}{array_hint}{_launch_line(plan, ddp)}\n"
    )


def _backup_target(path: Path) -> Path:
    """First free sibling `<name>.old`, `<name>.old2`, ... for *path*."""
    candidate = Path(str(path) + ".old")
    n = 2
    while candidate.exists():
        candidate = Path(f"{path}.old{n}")
        n += 1
    return candidate


def write_sbatch(
    project_root: Path, plan: RunPlan, sbatch_dir: str = "scripts", ddp: bool = False
) -> Path:
    """Write the rendered script to <root>/<sbatch_dir>/<entry-stem>.sbatch, backing up existing."""
    path = project_root / sbatch_dir / f"{Path(plan.entry).stem}.sbatch"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.rename(_backup_target(path))
    path.write_text(render_sbatch(plan, ddp), encoding="utf-8")
    path.chmod(0o755)
    return path
