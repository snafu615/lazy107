# failures/ — deliberate-failure harvest projects

Each numbered folder is a minimal project that fails in exactly one way when
submitted through `lazy107`. Purpose: capture the real log signature of each
failure class on the 107 cluster so the future `debug` command's signature
library matches reality.

These projects reuse the `demo` conda env from the happy-path runbook smoke —
no `lazy107 env` step, no dependency downloads.

## Prerequisites (do once)

- `lazy107 env --yes` has been run in the happy-path demo project and
  `conda env list` shows `demo`.
- `lazy107 discover` has been run (global account/partition/QoS config).
- This folder is on the login node (GUI upload supports folders).

## Which to run live (trimmed matrix)

| Project | Run live? | Why |
|---|---|---|
| 01-missing-module | **yes** | cheapest failure; validates the whole pipeline (submit → ledger → log → sacct) |
| 02-missing-data | no | traceback format is version-stable; synthesize locally |
| 03-syntax-error | no | same |
| 04-import-error | no | same |
| 05-time-limit | **yes** | capture 107's exact `slurmstepd ... DUE TO TIME LIMIT` wording |
| 06-oom-kill | **yes** | capture 107's exact oom-kill line wording |
| 07-cuda-oom | **yes** | capture torch CUDA OOM wording of the demo env's torch build |
| 08-cuda-guard | optional | trips lazy107's own guard (message already known); validates it fires on real hardware |

Run sequentially (P107 QoS: `MaxJobsPU=4`); each job finishes in a few minutes.

## Run steps (projects 01-07)

```bash
cd ~/failures/01-missing-module
lazy107 plan                  # sanity: gpu=1, partition=P107-RTX5090, conda_env=demo
lazy107 submit --yes          # prints "submitted job <job_id>"
lazy107 watch <job_id>        # or poll: squeue -j <job_id>
sacct -j <job_id> -o JobID,State,ExitCode,Elapsed --parsable2
cp logs/*.out ../fixtures/01-missing-module.log
```

## Project 08 (special: needs a hand-edit + manual sbatch)

`submit` re-renders the sbatch, so a manual edit does not survive it:

```bash
cd ~/failures/08-cuda-guard
lazy107 render                 # writes scripts/train.sbatch (no --dry-run)
# insert ONE line just before the guard heredoc (`python - <<'PY'` near the end):
#     export CUDA_VISIBLE_DEVICES=""
sbatch scripts/train.sbatch     # manual submit (no ledger entry; expected)
sacct -j <job_id> -o JobID,State,ExitCode,Elapsed --parsable2
cp logs/*.out ../fixtures/08-cuda-guard.log
```

## Expected signatures

| Project | Log signature | sacct ExitCode |
|---|---|---|
| 01 | `ModuleNotFoundError: No module named 'nonexistent_module_xyz'` | `1:0` |
| 02 | `FileNotFoundError: [Errno 2] No such file or directory: 'data/raw.pt'` | `1:0` |
| 03 | `SyntaxError: invalid syntax` | `1:0` |
| 04 | `ImportError: cannot import name 'resnet18' from 'model'` | `1:0` |
| 05 | `slurmstepd: error: *** JOB ... DUE TO TIME LIMIT ***` | `0:15` |
| 06 | `slurmstepd: error: Detected 1 oom-kill event(s)` (or an uncaught `MemoryError` traceback) | `0:9` |
| 07 | `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 GiB ...` | `1:0` |
| 08 | `AssertionError: GPU requested but CUDA unavailable in this job` | `1:0` |

## Caveats

- **Min limits**: if submit-time validation rejects `time = "0:02:00"` (05) or
  `mem = "4G"` (06) as below the QoS minimum, bump the value in `107.toml`
  and raise the sleep/allocation in `train.py` to match.
- **06 variability**: with cgroup enforcement the job may be SIGKILLed
  (oom-kill event) or numpy may raise `MemoryError` first. Either log is a
  valid harvest — the debug command should match both.
- Harvested logs go to `fixtures/`; pull them back down and later move them
  into `tests/fixtures/` as the debug command's test corpus.
