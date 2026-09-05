# Quick Start

Submit your first job on the USTC 107 cluster within 5 minutes. Everything
runs **on the cluster** (login node or Web Shell) — nothing is installed
locally.

## Prerequisites

- Access to the 107 login node (SSH) or Web Shell.
- Python 3.11+ available on the cluster for the tool itself.
- Code and data already on the cluster (see "Getting code and data onto the
  cluster" below).

### Getting code and data onto the cluster

lazy107 does not move files. Use any combination of:

- **git** (optional): `git clone` your repository on the login node. The tool
  never touches git; provenance lives in `notes/runs.md`.
- **GUI file manager** (SCOW / Xftp): drag-and-drop. For large data, run
  `lazy107 transfer` first — it prints a checklist:
  `tar -czf my-project-data.tar.gz ...` → upload → extract → `sha256sum`.

## Step 1: Install

```bash
# from a source checkout
pip install -e .

# or from a wheel
pip install lazy107-*.whl

lazy107 --version
```

### One-time platform setup

```bash
lazy107 discover --dry-run   # print what it resolved, write nothing
lazy107 discover             # write ~/.config/lazy107/config.toml
```

This resolves your account's allowed account/partition/QoS from the live
Slurm association and fixes "Invalid account or account/partition
combination specified": the tool's built-in default (`Students` /
`qos_stu_default`) is not authorized for competition accounts.

## Step 2: Scaffold a project (or use an existing one)

```bash
lazy107 init my-project
cd my-project
```

This creates: `src/data.py`, `src/model.py`, `src/train.py`,
`pyproject.toml` (with `torch`), `environment.yml`, `scripts/train.sbatch`,
`.gitignore`, `README.md`. Implement `src/data.py` and `src/model.py`, and put
your training loop in `src/train.py`.

For an existing project, lazy107 detects the entry file automatically:
`train.py` > `main.py` > first `*.py` in the tree.

### One command instead of steps 3-5

```bash
lazy107 everything
```

walks you through the whole pipeline with a prompt at each decision point:
pick the entry (when several exist), reuse an existing conda env (validated
against your imports) or install a fresh one, pick a Slurm resource preset
or adjust `gpu`/`cpus`/`mem`/`time` per field (Enter keeps the current
value), review the resolved plan, optionally
preview the sbatch script, and confirm the submit. `--yes` skips all prompts
(fresh env, current resources, direct submit); `--entry` preselects the
entry file.

## Step 3: Prepare the conda environment

```bash
lazy107 env --dry-run   # print first — this is always safe
lazy107 env --yes       # run it
```

For a scaffolded project this prints (roughly):

```bash
module load miniconda/py312  # no-op if conda is already on PATH
conda create -y -n my-project python=3.12
conda env update -n my-project -f environment.yml
conda run -n my-project pip install -e .
conda run -n my-project pip install torch torchvision  # CUDA wheel
```

Because the project depends on `torch` (GPU inferred), the pip install comes
from the default PyPI mirror — the default wheel bundles CUDA. If jax is used,
`jax[cuda12]` is installed instead; CPU-only builds are never the final
install for a GPU job.

A project that has no dependency files at all (just `.py` files) is handled
the same way: lazy107 scans the code for third-party imports, installs
each under its own name (`torch`, `numpy`, `requests`, ...; the few
mismatches like `sklearn` → `scikit-learn` are corrected), and writes them
to `requirements.txt` (so you can edit versions afterwards).
`env --dry-run` previews everything; `env --yes` is required to execute.

`env --yes` also wires the env to submissions: if `conda_env` is unset it
writes `conda_env = "my-project"` into a `107.toml` (created if missing),
so the rendered sbatch activates the right env automatically. If
`conda_env` is already set to a different name, it leaves it and tells you.

Already have a conda environment with the right packages? Run a plain
`lazy107 env` (no flags) in the terminal: it lists your existing
environments, validates your pick against what the project imports, and
pins it in `107.toml` — no reinstall. Press Enter to install a fresh
environment instead.

## Step 4: Plan and render

```bash
lazy107 plan
```

Expected output for a scaffolded project:

```text
entry=src/train.py
partition=Students
qos=qos_stu_default
account=
cpus=4
mem=16G
gpu=1
time=2:00:00
log_dir=logs
conda_env=
job_name=
```

(`partition`/`qos`/`account` reflect `~/.config/lazy107/config.toml` when
`lazy107 discover` has run.)

With several `.py` files in the project, `plan` shows a numbered picker —
Enter takes `train.py`, a number or filename picks another entry — and
pins the choice into `107.toml`. The last output line tells you exactly
which file holds every Slurm param: open the project folder in the Web
Shell GUI (Files), edit `107.toml`, and run `lazy107 plan` again to see
the result.

Then render without submitting:

```bash
lazy107 render --dry-run
```

Check that `--gres=gpu:1` is present (torch was detected) and that the script
contains the CUDA runtime assertion. Override anything via `107.toml` or
flags, e.g. `lazy107 render --dry-run` after setting `gpu = 0` in `107.toml`.

## Step 5: Submit

```bash
lazy107 submit --yes
```

`submit` runs the full pipeline:

1. validate the plan (entry exists, values safe);
2. `sinfo` preflight for the partition (skip with `--skip-check`);
3. if `gpu>0`: verify the env's torch build is CUDA, else **block**;
4. write `scripts/train.sbatch` (backing up any previous one);
5. `sbatch` the script, capture the job ID;
6. append the run to `notes/runs.md`.

**Acceptance**: output contains `submitted job <job_id>`.

## Step 6: Monitor

```bash
lazy107 watch <job_id>     # prints the four commands below
squeue -u $USER
scontrol show job <job_id>
tail -f logs/train_<job_id>.out
tail -f logs/train_<job_id>.err   # tracebacks land here
```

**Acceptance**: the job moves `PD` → `R` and training output appears in the
log.

## Step 6.5: Diagnose a failed job

```bash
lazy107 debug <job_id>
```

Combines the sacct exit code (standardized Slurm semantics: `0:9` SIGKILL,
`0:15` SIGTERM, `1:0` app error) with failure signatures from both log
halves, then prints a one-line cause and a fix:

```text
job 53073: state=FAILED exit=1:0
cause: missing dependency 'nonexistent_module_xyz'
fix: add nonexistent_module_xyz to requirements.txt/environment.yml and re-run `lazy107 env --yes`
```

## Acceptance checklist

- [ ] `lazy107 --version` works on the login node
- [ ] `lazy107 init my-project` created the expected files
- [ ] `lazy107 check` prints entry/env/runs status with a single `next:` hint
- [ ] `lazy107 plan` shows `gpu=1` for a torch project
- [ ] `lazy107 render --dry-run` prints a script with `--gres=gpu:1` and the
      CUDA assertion (GPU case), or neither (CPU case)
- [ ] `lazy107 submit --yes` returns `submitted job <job_id>`
- [ ] `squeue -u $USER` shows the job running
- [ ] `notes/runs.md` contains the job ID
- [ ] `lazy107 debug <job_id>` explains a failed job (cause + fix)
- [ ] `lazy107 transfer` prints a valid checklist (or "No large files found")
- [ ] `lazy107 everything` walks through env → resource presets → submit prompts

## Common blockers

### `env` says conda not found

```text
lazy107: conda not found: run `module load miniconda/py312` or install under ~/miniconda3
```

Run `module load miniconda/py312` in your shell, or install Miniconda under
`~/miniconda3`.

### Submit blocked on CPU-only torch

```text
GPU requested but the env's torch is a CPU-only build; reinstall via `lazy107 env --yes`
```

The env's torch was installed from a CPU source (e.g. conda-forge or a CPU
index). Reinstall with `lazy107 env --yes` — the PyPI default wheel bundles
CUDA.

### Submit failed: invalid account or account/partition combination

Your account cannot use the configured partition/QoS (e.g. the default
`Students`/`qos_stu_default` for a competition account). Fix: run
`lazy107 discover`, or set `account`/`partition`/`qos` in `107.toml`.

### `plan` / `render` fails with "entry" errors

No `train.py`/`main.py` found. Pass one explicitly:

```bash
lazy107 submit --entry src/other.py --yes
```

### Partition preflight fails

```text
partition <name> failed preflight; use --skip-check to override
```

The partition name is wrong or currently unavailable. Check with
`sinfo -p <name> -h`, fix `107.toml`, or use `--skip-check` deliberately.
