"""Deliberate failure #08: trip lazy107's own runtime CUDA guard.

The sbatch rendered by `lazy107 render` is hand-edited: one extra line
`export CUDA_VISIBLE_DEVICES=""` is inserted before the guard heredoc, then
submitted with plain `sbatch`. See ../README.md for the exact steps.

Expected log signature: AssertionError: GPU requested but CUDA unavailable in this job
sacct ExitCode: 1:0
"""

print("entry never runs: the guard aborts the job first")
