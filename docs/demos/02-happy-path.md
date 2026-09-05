# Demo 02 — Happy path end-to-end (1 GPU job)

What this proves: the entire pipeline works on a real cluster with a real
training result — entry detection, resource inference from *code only*, env
preparation with verified CUDA, deterministic sbatch rendering, submission,
monitoring, and the provenance ledger. Zero Slurm knowledge required.

Compute: one GPU job (~5 min on an RTX 5090) plus the one-time conda env
install (~10 min, reused by every later demo).

Capture: `./scripts/capture-demos.sh 02` -> `captures/02/*.txt`.

## Step 0: Get the example onto the cluster

`examples/resnet-cifar10/` is a code-only project: four `.py` files, no
`pyproject.toml`, no `requirements.txt`, no config. From the repo checkout:

```bash
cp -r examples/resnet-cifar10 ~/demos/demo
cd ~/demos/demo
```

(Or upload the folder through the SCOW GUI file manager — folder upload is
supported, no tar needed.)

## Step 1: The transfer checklist finds nothing (yet)

```bash
lazy107 transfer
```

```text
No large files found; nothing to transfer.
```

`transfer` prints a GUI upload checklist — it never syncs or deletes
anything. The example has no data yet, so the checklist is empty.

## Step 2: The plan infers everything from code

```bash
lazy107 plan
```

```text
entry=train.py
account=competition
partition=P107-RTX5090
qos=qos_p107-rtx5090
cpus=4
mem=16G
gpu=1
time=2:00:00
...
```

No dependency files exist — the tool scanned `train.py`/`data.py`/`model.py`
imports, found `torch`/`torchvision`/`tqdm`, and upgraded the plan to
`gpu=1, cpus=4, mem=16G, time=2:00:00`. The `account`/`partition`/`qos`
values come from the global config written by Demo 01's `discover`.

## Step 3: Environment, dry run

```bash
lazy107 env --dry-run
```

```text
# would write requirements.txt (imports: torch, torchvision, tqdm)
# would set conda_env = "demo" in 107.toml
module load miniconda/py312  # no-op if conda is already on PATH
conda create -y -n demo python=3.12
pip install -e .
conda run -n demo pip install torch torchvision tqdm  # PyPI default wheel bundles CUDA
```

A code-only project still gets a full environment: the imports become
`requirements.txt` (autowritten), and because `torch` was detected the final
install is pip from the default index — the PyPI wheel bundles CUDA, so the
env can never end on a CPU-only torch as the last install step.

## Step 4: Environment, for real (one time)

```bash
lazy107 env --yes
```

```text
wrote requirements.txt (imports: torch, torchvision, tqdm)
$ module load miniconda/py312  # no-op if conda is already on PATH
$ conda create -y -n demo python=3.12
... (install output) ...
torch CUDA build verified (torch.version.cuda=12.6)
wired conda_env = "demo" into 107.toml
```

Three things happened: the env was created from the USTC mirror, the torch
build was verified (`torch.version.cuda` non-empty — the required proof,
since `cuda.is_available()` is always False on the login node), and
`conda_env = "demo"` was wired into `107.toml` so every later submission
activates this env automatically. Re-running is idempotent (the create step
is skipped).

## Step 5: Render — dry run writes nothing

```bash
lazy107 render --dry-run
```

Key lines of the printed script:

```bash
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err
...
python - <<'PY'
import importlib.util
if importlib.util.find_spec('torch'):
    import torch
    assert torch.cuda.is_available(), 'GPU requested but CUDA unavailable in this job'
PY
```

`--gres=gpu:1` and the runtime CUDA assertion both appear because `gpu=1`.
The render is a pure function of the plan — the same inputs always produce
the same script, which is what makes `--dry-run` trustworthy.

## Step 6: Submit (preflight -> render -> confirm -> submit -> record)

```bash
lazy107 submit --yes
```

```text
submitted job 53080 (train)
watch: lazy107 watch 53080
```

Before `sbatch` ran, the pipeline validated the plan, checked the workflow
state, verified the partition exists (`sinfo`), verified the
account/partition/QoS association against live `sacctmgr`/`scontrol` data,
and verified the env's torch build is CUDA-capable. The job ID is captured
and immediately recorded.

## Step 7: Watch it run

```bash
lazy107 watch 53080
squeue -u $USER
```

The job goes `PD` -> `R` -> `COMPLETED`. `sacct` confirms:

```text
JobID   JobName  Partition     State      ExitCode  Elapsed
53080   train    P107-RTX5090  COMPLETED  0:0       00:04:57
```

## Step 8: Logs, and debug on a healthy job

```bash
lazy107 logs 53080
lazy107 debug 53080
```

```text
job 53080: state=COMPLETED exit=0:0
no failure detected (exit 0:0)
```

`logs` points at both halves (`logs/train_53080.out` for output,
`logs/train_53080.err` for tracebacks), and `debug` correctly reports
"nothing wrong" instead of hallucinating a diagnosis.

## Step 9: The result and the ledger

```bash
cat outputs/metrics.json
```

```text
{
  "best_test_acc": 0.78xx,
  "epochs": 5,
  "device": "cuda",
  "elapsed_s": 29x.x
}
```

A real ResNet-18 trained on CIFAR-10 on an RTX 5090, best checkpoint at
`outputs/best.pt`. And the provenance:

```bash
cat notes/runs.md
```

```markdown
## Run 53080

- submitted: 2026-09-03T...
- entry: train.py
- partition: P107-RTX5090
- qos: qos_p107-rtx5090
- cpus: 4, mem: 16G, gpu: 1, time: 2:00:00
```

## What this proves

- From four `.py` files to a trained model on a GPU node, with zero Slurm
  parameters written by hand.
- Resource inference works from code imports alone (no dependency files).
- The GPU/CUDA integrity chain: pip CUDA wheel -> `torch.version.cuda`
  verified at install -> submit-time build preflight -> runtime CUDA
  assertion in the job.
- Deterministic rendering, read-only dry runs, and a persisted ledger.
