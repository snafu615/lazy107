"""Tests for sbatch rendering and writing."""

import os
from pathlib import Path

from lazy107.core.plan import RunPlan
from lazy107.core.render import render_sbatch, write_sbatch


def test_render_sbatch_cpu() -> None:
    script = render_sbatch(
        RunPlan(entry="train.py", cpus=4, mem="16G", time="1:00:00", job_name="my_job")
    )

    assert "#SBATCH --job-name=my_job" in script
    assert "#SBATCH --partition=Students" in script
    assert "#SBATCH --qos=qos_stu_default" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=1:00:00" in script
    assert "#SBATCH --output=logs/%x_%j.out" in script
    assert "#SBATCH --error=logs/%x_%j.err" in script
    assert "--gres=gpu" not in script
    assert "--ntasks-per-node" not in script
    assert script.startswith("#!/bin/bash")
    assert "set -euo pipefail" in script
    assert 'cd "$SLURM_SUBMIT_DIR"' in script
    assert "mkdir -p logs" in script
    assert "python train.py" in script
    assert "torchrun" not in script
    # No conda block, no GPU guard for a plain CPU plan.
    assert "conda activate" not in script
    assert "torch.cuda.is_available" not in script


def test_render_sbatch_gpu() -> None:
    script = render_sbatch(RunPlan(entry="train.py", gpu=1, time="2:00:00", job_name="gpu_job"))

    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --job-name=gpu_job" in script
    assert "torch.cuda.is_available" in script
    assert "torchrun" not in script  # no DDP signal, single GPU: plain python


def test_render_ddp_single_node_uses_torchrun() -> None:
    script = render_sbatch(RunPlan(entry="train.py", gpu=2, nodes=1), ddp=True)

    assert "#SBATCH --gres=gpu:2" in script
    assert "--ntasks-per-node" not in script
    assert "torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE train.py" in script
    assert "python train.py" not in script


def test_render_ddp_multi_node_uses_srun_rdzv() -> None:
    script = render_sbatch(RunPlan(entry="train.py", gpu=4, nodes=2, ntasks=2), ddp=True)

    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks=2" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "srun torchrun --nnodes=$SLURM_JOB_NUM_NODES" in script
    assert "--nproc_per_node=$SLURM_GPUS_ON_NODE" in script
    assert "--rdzv_id=$SLURM_JOB_ID" in script
    assert '--rdzv_endpoint="$(scontrol show hostname $SLURM_NODELIST | head -n1):29500"' in script
    assert "python train.py" not in script


def test_render_ddp_without_gpu_stays_python() -> None:
    script = render_sbatch(RunPlan(entry="train.py", gpu=0), ddp=True)
    assert "python train.py" in script
    assert "torchrun" not in script


def test_render_command_override() -> None:
    plan = RunPlan(entry="train.py", command="python -m trainer.main --config cfg.yaml")
    script = render_sbatch(plan)

    assert "python -m trainer.main --config cfg.yaml" in script
    assert "python train.py" not in script


def test_render_command_wins_over_ddp() -> None:
    plan = RunPlan(entry="train.py", gpu=2, command="torchrun --nproc_per_node=2 train.py")
    script = render_sbatch(plan, ddp=True)

    assert "torchrun --nproc_per_node=2 train.py" in script
    assert "--standalone" not in script
    assert "python train.py" not in script


def test_render_array_job() -> None:
    script = render_sbatch(RunPlan(entry="train.py", array="1-5%2"))

    assert "#SBATCH --array=1-5%2" in script
    assert "#SBATCH --output=logs/%x_%A_%a.out" in script
    assert "#SBATCH --error=logs/%x_%A_%a.err" in script
    assert "# array task index available as $SLURM_ARRAY_TASK_ID" in script


def test_render_no_array_keeps_job_logs() -> None:
    script = render_sbatch(RunPlan(entry="train.py"))
    assert "--array" not in script
    assert "#SBATCH --output=logs/%x_%j.out" in script
    assert "%A_%a" not in script


def test_render_array_with_command_and_ddp() -> None:
    plan = RunPlan(entry="train.py", gpu=2, array="0,2,4")
    script = render_sbatch(plan, ddp=True)

    assert "#SBATCH --array=0,2,4" in script
    assert "torchrun --standalone" in script  # DDP launch per array task


def test_render_sbatch_conda() -> None:
    script = render_sbatch(
        RunPlan(entry="train.py", cpus=8, mem="32G", time="3:00:00", conda_env="myenv", job_name="conda_job")
    )

    assert 'source "$(conda info --base)/etc/profile.d/conda.sh"' in script
    assert "conda activate myenv" in script
    assert "set +u" in script
    assert "python train.py" in script
    assert ".venv/bin/python" not in script


def test_render_sbatch_default_job_name() -> None:
    assert "#SBATCH --job-name=train" in render_sbatch(RunPlan(entry="train.py"))


def test_render_sbatch_account_line_only_when_set() -> None:
    assert "--account=" not in render_sbatch(RunPlan(entry="train.py"))
    script = render_sbatch(RunPlan(entry="train.py", account="competition"))
    assert "#SBATCH --account=competition" in script


def test_write_sbatch_creates_file_and_backup(tmp_path: Path) -> None:
    plan = RunPlan(entry="train.py", job_name="test_job")
    path1 = write_sbatch(tmp_path, plan)
    assert path1.exists()
    if os.name != "nt":  # chmod exec bits are a POSIX concept
        assert path1.stat().st_mode & 0o111
    content1 = path1.read_text()
    assert "test_job" in content1

    path2 = write_sbatch(tmp_path, RunPlan(entry="train.py", job_name="test_job2"))
    assert path2 == path1
    backup = path1.with_name(path1.name + ".old")
    assert backup.exists()
    assert backup.read_text() == content1
    assert "test_job2" in path1.read_text()


def test_write_sbatch_custom_sbatch_dir(tmp_path: Path) -> None:
    path = write_sbatch(tmp_path, RunPlan(entry="train.py"), sbatch_dir="custom_scripts")
    assert path == tmp_path / "custom_scripts" / "train.sbatch"
    assert path.exists()
