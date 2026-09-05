# Demo 04 — debug: the tool explains failures (4 GPU jobs + 4 free)

What this proves: `lazy107 debug` turns a dead job into a one-line cause
and a one-line fix, matched against a signature library harvested from the
real 107 cluster — not generic advice. When a job dies, the tool explains
why.

Compute: four deliberate-failure jobs on `P107-RTX5090` (~2 min each,
sequential), reusing the `demo` env — no installs. The other three failure
classes (02/03/04) are matched locally at zero compute, because their
traceback formats are version-stable.

Capture: `./scripts/capture-demos.sh 04` -> `captures/04/*.txt`.
Reference: `examples/failures/SIGNATURES.md` (the live harvest results).

## Step 0: The failure projects

`examples/failures/` contains eight minimal projects, each failing in
exactly one way. We run four on the cluster:

| Project | Failure | sacct ExitCode |
|---|---|---|
| `01-missing-module` | `ModuleNotFoundError: No module named 'nonexistent_module_xyz'` | `1:0` |
| `05-time-limit` | sleeps past a 2-minute `--time` | `0:15` |
| `06-oom-kill` | allocates past `mem = 4G` | *(see Step 4 — the surprise)* |
| `07-cuda-oom` | allocates 40 GiB on a 32 GB RTX 5090 | `1:0` |

Each project points `conda_env = "demo"` at the env Demo 02 created, so
nothing is installed here.

## Step 1: Missing module

```bash
cd ~/demos/fail-01-missing-module
lazy107 submit --yes
# job FAILED after a few seconds
lazy107 debug <job_id>
```

```text
job 53082: state=FAILED exit=1:0
cause: missing dependency 'nonexistent_module_xyz'
fix: add nonexistent_module_xyz to requirements.txt/environment.yml and re-run `lazy107 env --yes`
evidence: ModuleNotFoundError: No module named 'nonexistent_module_xyz'
```

The signature came from the job's own `.err` log (real 107 traceback — see
`SIGNATURES.md`), and the fix is actionable: it names the exact package.

## Step 2: Wall-time limit

```bash
cd ~/demos/fail-05-time-limit
lazy107 submit --yes
# ~2 minutes later: CANCELLED
lazy107 debug <job_id>
```

```text
job 53083: state=TIMEOUT exit=0:15
cause: hit the wall-time limit
fix: raise `time` in 107.toml, add checkpointing, or reduce the workload
evidence: [2026-09-03T...] error: *** JOB 53083 ON anode01 CANCELLED AT ... DUE TO TIME LIMIT ***
```

Note the timestamped error format — that exact wording was captured live on
107 and became the matcher (`SIGNATURES.md` records that newer Slurm emits
`[ISO-ts] error:` without the classic `slurmstepd:` prefix).

## Step 3: CUDA out of memory

```bash
cd ~/demos/fail-07-cuda-oom
lazy107 submit --yes
lazy107 debug <job_id>
```

```text
job 53084: state=FAILED exit=1:0
cause: CUDA out of memory: tried to allocate 40.00 GiB but only 30.86 GiB free (GPU total 31.36 GiB)
fix: reduce batch size / model size, or use gradient accumulation
evidence: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 GiB. GPU 0 has a total capacity of 31.36 GiB of which 30.86 GiB is free
```

The matcher parses requested vs free memory out of torch's own OOM message
and says so in the cause line — the fix is specific to *this* failure, not
boilerplate.

## Step 4: The memory limit that didn't fire (a finding, not a bug)

```bash
cd ~/demos/fail-06-oom-kill
lazy107 submit --yes
lazy107 debug <job_id>
```

```text
job 53085: state=TIMEOUT exit=0:15
cause: terminated by SIGTERM (wall-time limit or manual cancel)
fix: raise `time` in 107.toml, add checkpointing, or reduce the workload
```

The harvest expected the OOM killer (`Detected N oom-kill event(s)`, exit
`0:9`). The job instead survived past its memory request and was killed by
the *time* limit — meaning `--mem=4G` was not enforced on
`P107-RTX5090`. The demo is honest about it: `debug` falls back to the
standardized sacct exit-code layer and still explains the state correctly,
and `SIGNATURES.md` records the open platform question. Diagnosing your
*platform*, not just your code.

## Step 5: The remaining signatures, matched at zero compute

The other three classes (missing data, runtime syntax error, wrong import)
produce version-stable tracebacks, so they can be verified without queueing
jobs:

```bash
python - <<'PY'
from lazy107.core.diagnose import diagnose
for text, exit_code in [
    ("FileNotFoundError: [Errno 2] No such file or directory: 'data/raw.pt'", "1:0"),
    ("SyntaxError: invalid syntax", "1:0"),
    ("ImportError: cannot import name 'resnet18' from 'model'", "1:0"),
]:
    d = diagnose(text, exit_code)
    print(f"cause: {d.cause}\nfix:   {d.fix}\n")
PY
```

```text
cause: missing data file 'data/raw.pt'
fix:   upload it to the cluster (see `lazy107 transfer`)

cause: SyntaxError: invalid syntax
fix:   check the traceback above

cause: cannot import 'resnet18' from 'model'
fix:   check the import name in the entry file
```

## What this proves

- `debug` explains failures in one line each of cause + fix + evidence,
  matched against log wording harvested from the *real* cluster
  (timestamped time-limit line, torch's OOM wording, traceback formats).
- The diagnosis has two layers: log signatures first, then standardized
  sacct exit-code semantics — so even unseen failures get a correct
  explanation (Step 4).
- The harvest process itself discovered a platform fact (mem not enforced
  on P107-RTX5090) — the tool makes the cluster observable.
