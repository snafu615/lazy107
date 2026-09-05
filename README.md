# lazy107

lazy107 is a cluster-side CLI for the USTC 107 computing platform. It detects
your entry file, infers resources (GPU/CPU/memory/time) from your dependencies,
prepares a conda environment, generates the sbatch script, submits the job, and
records every run — no Slurm parameters to memorize, nothing installed locally.

## Install

The single-file installer needs no network and no repo clone — upload
`dist/lazy107-0.1.0-install.sh` to the login node and run:

```bash
bash lazy107-0.1.0-install.sh
# -> installs into the active conda env; falls back to a dedicated conda
#    env "lazy107", then a venv at ~/.lazy107
lazy107 --version
```

Or install from source (`python >= 3.11` only, no other dependencies):

```bash
git clone <this-repo> lazy107 && cd lazy107
python -m pip install -e .          # dev
# or: python -m pip install dist/lazy107-0.1.0-py3-none-any.whl
```

## Quick start

```bash
# 1. On the 107 login node / Web Shell: one-time platform discovery
#    (resolves your account/partition/QoS from the live Slurm association)
lazy107 discover --dry-run
lazy107 discover

# 2. Scaffold a project
lazy107 init my-project
cd my-project

# ... or skip steps 3-4 and run the guided pipeline (env reuse or install,
# slurm presets, submit):
lazy107 everything

# 3. Prepare the conda environment (print-only by default; --yes to run)
lazy107 env --dry-run

# 4. See what will be submitted, then submit
lazy107 plan
lazy107 render --dry-run
lazy107 submit --yes

# 5. Monitor
lazy107 watch <job_id>     # prints squeue / scontrol / tail commands
squeue -u $USER
tail -f logs/train_*.out
tail -f logs/train_*.err   # tracebacks land here

# 6. Diagnose a failed job
lazy107 debug <job_id>    # sacct exit code + log signatures -> cause/fix
```

First time? Read [docs/quickstart.md](docs/quickstart.md). End-to-end worked
example: [docs/demo.md](docs/demo.md). Platform details and troubleshooting:
[docs/runbook.md](docs/runbook.md). Internal design:
[docs/design.md](docs/design.md).

## Highlights

- **Cluster-only**: runs on the 107 login node / Web Shell. Nothing to install
  locally — code and data move via git (optional) and the GUI file manager.
- **Smart defaults**: `torch`/`tensorflow`/`jax` in your dependencies →
  GPU/4 CPU/16G/2h; otherwise CPU/2 CPU/4G/1h. Override in `107.toml`, via
  `LAZY107_*` environment variables, or CLI flags.
- **Code-only projects work**: no dependency files? `env` scans your `.py`
  imports (torch, numpy, sklearn→scikit-learn, ...), writes them to
  `requirements.txt`, and installs from there. `env` runs only with `--yes`;
  `--dry-run` previews everything including the generated manifest.
- **Reuse environments, skip reinstall**: a plain `lazy107 env` in the
  terminal lists your existing conda envs; picking one validates it against
  the project's dependencies (importability probe, plus the CUDA build check
  for GPU jobs) and pins it as `conda_env` in `107.toml`. Enter installs fresh.
- **GPU/CUDA integrity, enforced**: `gpu>0` can never silently resolve to a
  CPU-only build — install-time pinning, post-install build verification,
  preflight blocking, and a runtime CUDA assertion in the generated script.
- **All conda, no uv**: environments are `conda create -n <project> python=3.12`
  + pip installs; batch activation uses the documented
  `source "$(conda info --base)/etc/profile.d/conda.sh"` pattern.
- **Deterministic scripts**: `RunPlan` is the single source of truth; every
  sbatch line is explicit (partition, QoS, account, CPU, mem, GPU, time,
  `--nodes=1`, logs).
- **Per-user discovery**: `lazy107 discover` resolves the account/partition/
  QoS your Slurm account is actually allowed to use (verified via
  `sacctmgr`/`scontrol`) into `~/.config/lazy107/config.toml` — no more
  "Invalid account or account/partition combination" surprises at submit
  time.
- **Safe by design**: `--dry-run` never writes or submits; `transfer` only
  *prints* a GUI upload checklist (tar → upload → extract → sha256) — no
  rclone, no Pan, no automation of destructive operations.
- **Provenance**: every successful submit is appended to `notes/runs.md`
  (job ID + effective parameters) and every command run inside the project
  is appended to `notes/history.md` (so you can always see what you already
  did; `lazy107 check` shows the count and latest) — git stays optional.

## Installation (cluster side)

lazy107 is a plain Python package with zero runtime dependencies.

```bash
# from a source checkout
pip install -e .

# or from a built wheel
pip install lazy107-*.whl
```

Verify:

```bash
lazy107 --version
```

## Subcommands

| Command | Purpose |
|---|---|
| `lazy107 init <name>` | Scaffold a new project from the bundled template |
| `lazy107 plan [--entry] [--array]` | Print the resolved run plan; asks which entry when several exist and pins the choice in `107.toml` |
| `lazy107 render [--entry] [--array] [--dry-run]` | Write (or print) `scripts/<entry>.sbatch` |
| `lazy107 env [--name] [--entry] [--dry-run] [--yes]` | Print/run conda env preparation with GPU pinning; interactively offers reuse of an existing validated env |
| `lazy107 submit [--entry] [--array] [--dry-run] [--skip-check] [--yes]` | Validate → preflight → render → confirm → submit → record |
| `lazy107 watch <job_id> [--job-name]` | Print monitoring commands (squeue/scontrol/tail) |
| `lazy107 status [job_id]` | Print queue status commands |
| `lazy107 logs <job_id> [--job-name]` | Print the tail commands (stdout + stderr logs) |
| `lazy107 debug <job_id> [--job-name]` | Diagnose a failed job: sacct exit code + log signatures → cause/fix |
| `lazy107 transfer` | Print the large-file GUI upload checklist |
| `lazy107 discover [--dry-run]` | Resolve your account/partition/QoS and write the per-user global config |
| `lazy107 config [--init]` | Show the effective config, or write a `107.toml` |
| `lazy107 check` | Show workflow status (entry/env/runs) and the next step |
| `lazy107 everything [--entry] [--yes]` | Guided end-to-end: entry → env (reuse or install) → Slurm presets (or per-field edit) → plan → optional preview → submit |

Workflow guards: `submit` blocks loudly when `conda_env` is configured but the
env was never created, and warns about skipped steps (no env wired, `watch`
before submit, unknown job ids).

## Configuration

Four layers, later wins:

1. **Tool defaults + dependency inference** (GPU/CPU resources from
   `pyproject.toml` / `requirements.txt` / `environment.yml`)
2. **Per-user global config** `~/.config/lazy107/config.toml` (written by
   `lazy107 discover`; holds `account`/`partition`/`qos`)
3. **Project `107.toml`** (flat keys or a `[run]` section; create with
   `lazy107 config --init`)
4. **`LAZY107_*` environment variables and CLI flags**

```toml
account = ""              # empty → Slurm default account
partition = "Students"
qos = "qos_stu_default"
cpus = 4
mem = "16G"
gpu = 1
nodes = 1          # >1 → multi-node allocation
ntasks = 1         # multi-node DDP: must equal nodes
time = "2:00:00"
log_dir = "logs"
conda_env = "my-project"
job_name = ""
command = ""        # set → launch verbatim instead of python/torchrun
array = ""          # set → Slurm job array, e.g. 1-5%2 or 0,2,4
```

`lazy107 plan` asks which entry to use when the project has several `.py`
files (Enter takes the `train.py` recommendation) and pins the choice as
`entry` here, so every later command honors it. Edit any value directly in
the file — open the project folder in the Web Shell GUI (Files) or any
terminal editor — then run `lazy107 plan` again to see the effect.

## Resource presets (wizard)

`lazy107 everything` offers named Slurm resource presets instead of a
measurement step: `cpu-light` (the CPU default), `cpu-heavy`, `gpu-light`
(the GPU default), `gpu-heavy`, and `gpu-multi` (2 GPUs). Enter keeps the
plan's current values; picking a preset wires all four keys (`gpu`/`cpus`/
`mem`/`time`) into `107.toml` at once; `c` falls through to the per-field
prompts (each printed with common-value hints). Values equal to the current
plan are dropped, so nothing is rewritten pointlessly.

```text
slurm resources (Enter keeps the current values):
  1) cpu-light  gpu=0 cpus=2 mem=4G time=1:00:00
  2) cpu-heavy  gpu=0 cpus=8 mem=32G time=8:00:00
  3) gpu-light  gpu=1 cpus=4 mem=16G time=2:00:00
  4) gpu-heavy  gpu=1 cpus=8 mem=64G time=12:00:00
  5) gpu-multi  gpu=2 cpus=16 mem=64G time=24:00:00
pick a preset [1-5], c to customize each field: 
```

Presets are honest starting points, not predictions. Start small, submit,
and bump on evidence: `lazy107 debug <job_id>` turns an OOM kill
(`0:9`) into "raise mem" and a timeout (`0:15`) into "raise time" — then
raise that one value (directly in `107.toml` or through the wizard) and
resubmit. `lazy107 plan` always shows what will take effect.


## Safety boundaries

- **Never train on the login node.** Everything goes through `sbatch`.
- **`--dry-run` is read-only** — it prints, it never writes or submits.
- **`transfer` prints commands only** — no automated sync, no deletions.
- **GPU means CUDA.** A GPU request is verified end-to-end (see
  [docs/design.md §8](design.md)).
- **No secrets in files.** No tokens, passwords, or credentials in configs,
  scripts, logs, or notes.

## Documentation

| Doc | Audience | Contents |
|---|---|---|
| [docs/quickstart.md](docs/quickstart.md) | First-time users | 5 minutes from install to first job |
| [docs/design.md](docs/design.md) | Developers, maintainers | Architecture, CLI, config system, GPU/CUDA integrity, safety |
| [docs/demo.md](docs/demo.md) | Anyone curious | End-to-end worked example |
| [docs/demos/INDEX.md](docs/demos/INDEX.md) | Evaluators, first-time users | Six live demo walkthroughs (install, happy path, wizard presets, debug, DDP/arrays, guards) |
| [docs/runbook.md](docs/runbook.md) | Acceptance testing | Step-by-step, acceptance criteria, common blockers |

## License

MIT License.
