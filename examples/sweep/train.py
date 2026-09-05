"""Synthetic hyperparameter sweep entry for a Slurm job array.

lazy107 renders the array spec from 107.toml (`array = "1-4%2"`) into the
sbatch script; Slurm then runs this entry once per task with
$SLURM_ARRAY_TASK_ID set (1..4). Each task picks its grid row, "trains" for
a few steps, and prints its own progress — the per-task logs demonstrate the
array-aware `%x_%A_%a` log naming (`logs/sweep_<id>_<task>.out`).

Runs standalone too: without the Slurm variable the first row is used, so
you can try it locally with `python train.py`.
"""

import os
import time

# (learning rate, steps) — one row per array task, in task order.
GRID = [(0.001, 3), (0.003, 3), (0.01, 3), (0.03, 3)]


def main() -> None:
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    lr, steps = GRID[task - 1]
    print(f"task {task}/{len(GRID)}: lr={lr} steps={steps}")
    loss = 1.0
    for step in range(1, steps + 1):
        time.sleep(2)  # stand-in for one training step
        loss *= 0.5
        print(f"  step {step}/{steps}: simulated_loss={loss:.4f}")
    print(f"task {task} done: final_loss={loss:.4f}")


if __name__ == "__main__":
    main()
