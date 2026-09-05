# Demo 06 — Guard rails (1 GPU job + zero-compute blocks)

What this proves: the safety invariants are enforced, not documented. A
misconfigured workflow is blocked *before* anything is queued; the confirm
prompt is honored; config layers behave exactly as specified; and a GPU
request can never silently become CPU training — the generated script
aborts the job if CUDA is missing.

Compute: one ~1-minute GPU job (the guard trip); everything else is
zero-compute blocks on the login node.

Capture: `./scripts/capture-demos.sh 06` -> `captures/06/*.txt`.

## Part A — The workflow guard blocks a doomed submit

### Step 0: A project whose env was never created

```bash
cd ~/demos
lazy107 init guard
cd guard
lazy107 config --init
# add: conda_env = "guard-never-created"
```

`guard` is a fresh scaffold; its `107.toml` now points at a conda env that
does not exist. The rendered script would `conda activate` it and die
instantly — so the tool refuses:

```bash
lazy107 submit --yes
```

```text
lazy107: conda env 'guard-never-created' is configured in 107.toml but not created; run `lazy107 env --yes` before submitting
```

Exit code 1, no sbatch written, nothing queued. The check happens before
the partition preflight — the mistake never reaches Slurm. (Skipped steps
that *can* work instead print loud warnings; blocking is reserved for
deterministic failures.)

## Part B — The confirm prompt is honored

### Step 1: Answering `n` produces nothing

In the working project from Demo 02 (env ready, preflight passes):

```bash
cd ~/demos/demo
echo n | lazy107 submit
```

```text
submit train on P107-RTX5090 (gpu=1)? [y/N] lazy107: aborted
```

No sbatch write, no submission, no ledger entry — the answer is honored
(and in a non-interactive shell, `submit` aborts cleanly instead of
assuming "yes").

## Part C — Config layers behave as documented

### Step 2: `gpu = 0` removes the GPU request — and the guard

```bash
cd ~/demos/demo
# temporarily add to 107.toml:
#   gpu = 0
lazy107 render --dry-run
```

No `--gres=gpu:1` line, and the CUDA assertion block is gone. Same plan
mechanism, opposite decision — deterministic.

### Step 3: An env var overrides the manifest

```bash
LAZY107_GPU=2 lazy107 plan
```

```text
gpu=2
```

Manifest says 0, env says 2, plan shows 2 — the documented precedence
(defaults < inference < global config < `107.toml` < `LAZY107_*` env < CLI
flags) is observable, not aspirational. (`107.toml` is restored afterwards.)

## Part D — The runtime CUDA guard fires on real hardware

### Step 4: Hide the GPU from a job that asked for one

`examples/failures/08-cuda-guard` requests `gpu = 1`. We render the script,
hand-edit one line in front of the guard (`export CUDA_VISIBLE_DEVICES=""` —
the equivalent of a misconfigured environment hiding the GPU), and submit
manually:

```bash
cd ~/demos/fail-08
lazy107 render
# insert before the guard heredoc:  export CUDA_VISIBLE_DEVICES=""
sbatch scripts/train.sbatch
```

The job fails within seconds:

```text
job 53087: state=FAILED exit=1:0
cause: AssertionError: GPU requested but CUDA unavailable in this job
fix: check the traceback above
evidence: AssertionError: GPU requested but CUDA unavailable in this job
```

The entry never ran — the guard aborted the job before a single training
step could silently execute on the CPU. (The submit-time preflight in the
normal flow would have caught this before queueing; the manual `sbatch`
bypasses it, which is exactly the scenario the runtime guard exists for.)

Note the ledger-aware warning too: `debug` flags that this manual job is
not in `notes/runs.md` — provenance gaps are surfaced, not ignored.

## What this proves

- Wrong-order and skipped steps are caught loudly: blocking for
  deterministic failures, warnings otherwise.
- `--dry-run` purity, honored confirm prompts, and the documented config
  precedence are all observable behavior, not claims.
- The GPU/CUDA integrity chain has four enforcement points; this demo trips
  the last one — the runtime assertion inside the generated script — on
  real hardware.
