# Demo: End-to-End Submission Flow

> For the live demo series run on the cluster (install + discover, happy
> path, wizard presets, debug, DDP/arrays, guards), see
> [docs/demos/INDEX.md](demos/INDEX.md).

This walkthrough uses a scaffolded project `demo` and records the flow from
`lazy107 init` to a rendered, submittable script. Local steps were run and
verified on 2026-08-22 (Windows dev machine, `python -m lazy107`); cluster
steps are marked **live** and complete the runbook's Step 7-8 once run on the
107 login node.

> Timings are only given for the local smoke; cluster times depend on the
> queue and mirror load.

## Scenario

- Project: `demo` (scaffolded with `lazy107 init`)
- Entry: `src/train.py`
- Dependencies: `torch`, `torchvision`, `tqdm` (pyproject.toml)
- Partition: `Students`, QoS: `qos_stu_default`
- Resources: 1 GPU / 4 CPU / 16G / 2:00:00 (inferred)

## Step 1: Scaffold

```bash
lazy107 init demo
cd demo
```

```text
created demo
```

```bash
tree -L 2   # (or: find . -maxdepth 2 -type f)
```

```text
.
├── environment.yml
├── pyproject.toml
├── .gitignore
├── README.md
└── src
    ├── __init__.py
    ├── data.py
    ├── model.py
    └── train.py
```

(`scripts/train.sbatch` is also created inside `demo/scripts/`.)

## Step 2: Plan

```bash
lazy107 plan
```

```text
entry=src/train.py
partition=Students
qos=qos_stu_default
cpus=4
mem=16G
gpu=1
time=2:00:00
log_dir=logs
conda_env=
job_name=
```

`torch` in pyproject.toml → `gpu=1` and the GPU resource set. (A project
without a GPU framework resolves to `gpu=0`, `cpus=2`, `mem=4G`,
`time=1:00:00`.)

## Step 3: Environment (dry-run)

```bash
lazy107 env --dry-run
```

```text
module load miniconda/py312  # no-op if conda is already on PATH
conda create -y -n demo python=3.12
conda env update -n demo -f environment.yml
conda run -n demo pip install -e .
conda run -n demo pip install torch torchvision  # PyPI default wheel bundles CUDA
```

Manifest installs are additive (`environment.yml` **and** `pyproject.toml`
are both honored), and because `torch` was detected the final install is
pip from the default PyPI mirror — the default wheel bundles CUDA, so the
env can never end with a CPU-only torch as the last install step. A
code-only project (no dependency files) gets the same treatment: dry-run
previews `# would write requirements.txt (imports: ...)`, and `env --yes`
writes that file before installing. **Live**: run `lazy107 env --yes`,
then verify with
`conda run -n demo python -c "import torch; print(torch.version.cuda)"` —
a non-empty version is the required proof (see design.md §8). `env --yes`
also wires `conda_env = "demo"` into a minimal 107.toml, so every later
submission activates this env.

## Step 4: Render (dry-run)

```bash
lazy107 render --dry-run
```

Key lines of the printed script:

```bash
#!/bin/bash
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%x_%j.out
...
if [ "$LAZY107_GPU" = "1" ]; then
  python - <<'PY' ... import torch; sys.exit("GPU requested but CUDA unavailable in this job") if not torch.cuda.is_available() ...
PY
fi
```

Both `--gres=gpu:1` and the CUDA assertion appear because `gpu=1`. With
`gpu = 0` in `107.toml`, neither appears. `--dry-run` writes nothing.

## Step 5: Submit (dry-run → confirm)

```bash
lazy107 submit --dry-run --yes
# prints the same script, writes nothing, returns 0

lazy107 submit --yes
```

```text
submitted job 1234567
watch: lazy107 watch 1234567
```

Before the real submit, `submit` also runs:

1. plan validation + entry resolution;
2. `sinfo -p Students -h` preflight (skip with `--skip-check`);
3. GPU preflight: env torch build check (blocks on CPU-only builds);
4. sbatch write (previous `scripts/train.sbatch` backed up);
5. `sbatch` capture of job ID;
6. `notes/runs.md` append:

```markdown
## Run 1234567

- submitted: 2026-08-22T...
- entry: src/train.py
- partition: Students
- qos: qos_stu_default
- cpus: 4, mem: 16G, gpu: 1, time: 2:00:00
```

Without `--yes`, the tool asks for confirmation and honors the answer —
answering `n` produces no artifacts at all.

## Step 6: Monitor

```bash
lazy107 watch 1234567
```

```text
squeue -u <user>
scontrol show job 1234567
tail -f logs/train_1234567.out
tail -f logs/train_1234567.err   # tracebacks land here
```

**Live**: watch the job go `PD` → `R`, training output appear, and the job
complete with exit code 0. If it fails instead,
`lazy107 debug 1234567` prints a one-line cause and fix (e.g. missing
dependency → re-run `lazy107 env --yes`).

## Step 7: Transfer (data upload checklist)

```bash
lazy107 transfer
```

With `data/raw.bin` present:

```text
1. tar: tar -czf demo-data.tar.gz "data"
2. upload: upload demo-data.tar.gz via the GUI file manager (SCOW / Xftp) to ~/transfer/
3. extract: tar -xzf demo-data.tar.gz -C ~/transfer/
4. verify: sha256sum demo-data.tar.gz
```

Output only — nothing is executed, and the archive itself is excluded from
its own checklist.

## Verification summary

| Step | Verified |
|---|---|
| init scaffolds expected files | ✅ local |
| plan infers gpu=1 from torch | ✅ local |
| env --dry-run prints conda sequence, pip last | ✅ local |
| render --dry-run: --gres + CUDA guard, nothing written | ✅ local |
| submit --dry-run never submits/writes | ✅ local |
| transfer prints-only checklist | ✅ local |
| env --yes real install, CUDA build check | ⏳ live (runbook Step 3/8) |
| sbatch submit → job R → COMPLETED | ⏳ live (runbook Step 7) |
| qos_stu_default / mirror facts | ⏳ live (runbook Step 8) |
