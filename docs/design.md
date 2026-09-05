# lazy107 Design Document

This document describes the architecture, CLI structure, configuration
system, GPU/CUDA integrity guarantees, and safety boundaries of lazy107.

## 1. Positioning

lazy107 is a **cluster-only** command-line tool for submitting training jobs
to the USTC 107 Slurm cluster. It runs entirely on the 107 login node /
Web Shell: one binary, no local tooling, no remote dependency on cloud
storage.

Design decisions (all locked with the user):

| Decision | Consequence |
|---|---|
| Cluster-only | Zero local install; code/data reach the cluster via git (optional) and the GUI file manager |
| All-conda, no uv | Env preparation = `conda create -n <project> python=3.12` + pip installs; batch activation via `conda.sh` sourcing |
| No Pan/rclone | `transfer` prints a GUI upload checklist (tar → upload → extract → sha256); nothing is automated |
| Git optional | Never invoked by the tool; provenance lives in `notes/runs.md` |
| Config-file-first, no wizard | `~/.config/lazy107/config.toml` (per-user, from `discover`) + project `107.toml` + `LAZY107_*` env vars + CLI flags; four layers, later wins |
| Zero runtime dependencies | stdlib only: `tomllib`, `importlib.resources`, `dataclasses`, `subprocess` |

## 2. Typical workflow

```text
[login node]
  lazy107 discover           one-time: resolve account/partition/QoS, write global config
  lazy107 init my-project      scaffold (or bring an existing project)
  lazy107 env --dry-run        print conda preparation
  lazy107 env --yes            create env, install deps (GPU-pinned)
  lazy107 plan                 show resolved run plan
  lazy107 render --dry-run     show the sbatch script (read-only)
  lazy107 submit --yes         validate → preflight → render → confirm → sbatch → record
  lazy107 watch <job_id>       print squeue / scontrol / tail commands (.out + .err)
  lazy107 debug <job_id>       diagnose a failed job: exit code + log signatures → cause/fix
  lazy107 transfer             print data upload checklist (when needed)
```

The tool never runs training itself; every job goes through `sbatch`. The
login node is for job management only.

## 3. CLI command structure

One entry point, fourteen subcommands (`lazy107 --help`):

| Command | I/O | Notes |
|---|---|---|
| `init <name>` | creates `./<name>` | scaffolds from the bundled template, `{project}` substitution |
| `plan [--entry] [--array]` | prints / pins | resolved `RunPlan`, one `key=value` per line; with several entry candidates and a TTY it shows a numbered picker (Enter = recommendation) and pins the choice as `entry` in `107.toml`; ends with a hint pointing at `107.toml` for GUI edits |
| `render [--entry] [--array] [--dry-run]` | writes / prints | `scripts/<entry>.sbatch`; non-dry-run backs up an existing file |
| `env [--name] [--entry] [--dry-run] [--yes]` | prints / runs | deterministic command sequence; GPU install pinning; post-install build check; on a TTY (no `--name`) offers reuse of an existing env, validated against the project's deps, and pins the pick as `conda_env` |
| `submit [--entry] [--array] [--dry-run] [--skip-check] [--yes]` | prints / submits | full pipeline, see §4 |
| `watch <job_id> [--job-name]` | prints | `scontrol show job`, `squeue --me`, `tail -f logs/<name>_<id>.{out,err}` |
| `status [job_id]` | prints | `squeue -j` / `sacct` variants |
| `logs <job_id> [--job-name]` | prints | tail commands for the job's `.out` and `.err` logs |
| `debug <job_id> [--job-name]` | prints | failure diagnosis: sacct exit code + log signatures → cause/fix (see §6) |
| `transfer` | prints | large-file GUI upload checklist; never executes anything |
| `discover [--dry-run]` | prints / writes | queries `sacctmgr`/`scontrol` for the user's allowed account/partition/QoS and writes the per-user global config; `--dry-run` prints only |
| `config [--init]` | prints / writes | effective config summary; `--init` writes `107.toml` |
| `check` | prints | workflow status (entry/env/runs) with a single `next:` step |
| `everything [--entry] [--yes]` | prints / prompts / runs | guided end-to-end: entry picker → env reuse-or-install → Slurm resource presets (or per-field editor) → plan → optional sbatch preview → submit; see §3.1 |

Exit codes: `0` success, `1` any failure (message on stderr prefixed
`lazy107:`). `main()` catches `ValueError`, `FileNotFoundError`,
`SubmitError`, and the validator's `PlanError` family, so a traceback is a
bug, not a user error.

### 3.1 The guided pipeline (`everything`)

`everything` is an external scaffold (`src/lazy107/wizard.py`, imported
lazily from `cli.build_parser`): it reuses the `cmd_*` functions by calling
them with `SimpleNamespace` args and the cli module's interactive helpers
(`_prompt_entry`, `_prompt_reuse_env`),
so no plan/env/submit logic is duplicated. Each decision point is a
plain prompt; `--yes` runs it non-interactively (fresh env, current
resources kept, direct submit). The Slurm editor offers named resource
presets (`PRESETS` in wizard.py: cpu-light/cpu-heavy/gpu-light/gpu-heavy/
gpu-multi) — Enter keeps the plan's current values, a numbered pick wires
all four keys at once (values equal to the current plan are dropped), and
`c` falls through to per-field prompts with common-value hints. The wizard
wires edits via `manifest.wire_fields` (ints for
gpu/cpus, strings for mem/time) and defers range/safety checks to
`validate_plan`, which runs in the plan print right after.

## 4. The submit pipeline

`lazy107 submit` is the core command. Order matters — each stage is
read-only until the last one:

1. **Resolve** — `_plan_for(cwd, entry)`: entry = `--entry` flag >
   `entry` pinned in `107.toml` > detection (train.py > main.py >
   first *.py), merge defaults < inference < manifest < env < flags.
   `lazy107 plan` performs the same resolution and, when the entry is
   neither flagged nor pinned and stdin is a TTY, interactively asks
   which entry to use and writes the choice into `107.toml`.
2. **Validate** — entry exists (`entry_path`), identifiers safe (no
   whitespace/shell metacharacters in entry/job_name/conda_env), partition
   and QoS non-empty, resources non-negative. `PlanError` → abort with a
   readable message.
3. **Notebook** — if the entry is an `.ipynb`, convert to a `.py` next to it
   (`nbconvert` lazily imported; interactive code stripped; a backup is
   written first).
4. **Workflow check** — `core/workflow.py`: if `conda_env` is configured but
   the env was never created, abort (the rendered `conda activate` would kill
   the job immediately; `--skip-check` bypasses). Loud stderr warnings for
   skipped steps: no env wired with third-party imports, env exists but
   `conda_env` unset. Fails open when conda is unavailable.
5. **Slurm preflight** — `sinfo -p <partition> -h` must succeed, unless
   `--skip-check` (needed because partition names are not queried live at
   plan time). Then the association check (`cluster/discover.py`): the
   effective account (plan `account`, else the Slurm default account) must
   be in the partition's `AllowAccounts` and the QoS must be both assigned
   to the account and in the partition's `AllowQos`; a mismatch aborts with
   a message naming the problem and pointing at `lazy107 discover`. The
   same check rejects `time`/`mem` that exceed the partition's
   `MaxTime`/`MaxMemPerNode` (compared only when both sides parse). The
   check fails open when `sacctmgr`/`scontrol` are unavailable or
   unparseable.
6. **GPU preflight** — if `gpu > 0`: locate conda, query the env's torch
   build (`torch.version.cuda`), and block if it is a CPU-only build
   (`gpu_preflight_error`). See §8.
7. **Render** — write `scripts/<entry>.sbatch` (backup of a previous file),
   or print it under `--dry-run`. **`--dry-run` short-circuits before any
   write or submit.**
8. **Confirm** — unless `--yes`, ask `submit job <id>? [y/N]` and honor the
   answer; `n` leaves no artifacts.
9. **Submit** — `sbatch` via `submit_sbatch`; any nonzero exit becomes a
   `SubmitError` with stderr captured (no traceback).
10. **Record** — append `## Run <job_id>` (timestamp, entry, partition, QoS,
   resources) to `notes/runs.md`, creating it if needed.

## 5. Directory structure

```text
src/lazy107/
├── cli.py               # argparse surface, all 14 commands, _fail/_entry helpers
├── wizard.py            # `everything` guided pipeline (orchestrates cli.cmd_*)
├── manifest.py          # config merge (4 layers) + global config path/loading + wire_* helpers
├── template.py          # bundled template resolution (importlib.resources)
├── transfer.py          # large-file discovery + print-only checklist
├── core/
│   ├── plan.py          # RunPlan dataclass (single source of truth)
│   ├── defaults.py      # tool defaults + dependency inference (pyproject/requirements/environment.yml)
│   ├── detect.py        # entry detection (train.py > main.py > first *.py)
│   ├── diagnose.py      # failure signature matching (pure, exit code + log text; see §6)
│   ├── validate.py      # plan validation + GPU preflight predicate
│   ├── workflow.py      # workflow-state checks (order/skips) + check dashboard
│   ├── render.py        # sbatch rendering (pure function of RunPlan)
│   ├── notebook.py      # .ipynb → .py conversion
│   ├── record.py        # notes/runs.md append
│   └── submit.py        # sbatch/sinfo execution, SubmitError
├── cluster/
│   ├── env.py           # conda discovery, env existence, command sequence, GPU pinning, build check
│   ├── discover.py      # sacctmgr/scontrol parsing, account/partition/QoS resolution + submit-time check
│   └── monitor.py       # watch/logs command printing, sacct lookup, log collection
└── templates/default/   # package data (wheel-safe): environment.yml, pyproject.toml,
                         #   .gitignore, README.md, scripts/train.sbatch, src/*.py

tests/                   # behavior spec, pytest, zero network
docs/                    # this documentation set
```

Layering rule: `core/*` is pure (no subprocess, no cluster), `cluster/*`
owns subprocess I/O, `cli.py` orchestrates. `render.py` is a pure function
of `RunPlan` — the same plan always produces the same script, which is what
makes `--dry-run` trustworthy.

## 6. Component responsibilities

### RunPlan (`core/plan.py`)

The single source of truth for a submission: `entry, partition, qos,
account, cpus, mem, gpu, nodes, ntasks, time, log_dir, conda_env,
job_name, command, array`.
`effective_job_name` is `job_name` or the entry stem — the sbatch
`--job-name` and the log filename always use the *same* value (defect #4:
job name and log path used to drift). `account` renders as
`#SBATCH --account=...` only when non-empty. `nodes`/`ntasks` (both
default 1) size the allocation; `command` (default empty) replaces the
generated launch line verbatim; `array` (default empty, Slurm `--array`
syntax like `1-5%2` or `0,2,4`) turns the job into a sweep.

### Defaults and inference (`core/defaults.py`)

Tool defaults: `Students` / `qos_stu_default` / `cpus=4` / `mem=16G` /
`gpu=0` / `time=1:00:00` / `log_dir=logs`.

Dependency inference reads `pyproject.toml`, `requirements.txt`, and
`environment.yml` (regex over `- name` lines, normalized `_`→`-`). A GPU
framework (`torch`/`pytorch`/`torchvision`/`torchaudio`/`tensorflow`/`jax`/
`jaxlib`) upgrades the plan to `gpu=1, cpus=4, mem=16G, time=2:00:00`;
otherwise CPU defaults apply (`gpu=0, cpus=2, mem=4G, time=1:00:00`). The
two profiles are deliberate conservative defaults — **no workload
estimation** (batch size, epochs, model size, workers) is attempted;
projects that know their needs override them in `107.toml`. Inference
never overrides an explicit
`107.toml` value (the old `_explicit_fields` semantics are preserved).

`detect_ddp()` scans the same code files for high-confidence distributed
training signals (torch.distributed imports, `DistributedDataParallel(...)`,
`init_process_group(...)`, accelerate/deepspeed imports, `strategy="ddp"`,
`ddp_find_unused_parameters=`). Comments and `DataParallel` never trigger
it — a false positive would launch the entry N times under torchrun.

### Slurm render (`core/render.py`)

`render_sbatch(plan, ddp)` is a deterministic string. Directives: job-name,
account (when set), partition, qos, nodes, ntasks, cpus-per-task, mem, gres
(when gpu>0), time, output/error; the script `mkdir -p`s `log_dir` before
running. Launch line, first match wins:

1. `command` (when set) — verbatim, full user control;
2. DDP GPU job — `torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE
   <entry>` on one node; on multiple nodes `--ntasks-per-node=1` plus the
   PyTorch-documented `srun torchrun` c10d rendezvous recipe (validation
   requires `ntasks == nodes` there);
3. `python <entry>` otherwise.

When `array` is set: the script emits `#SBATCH --array=<spec>` (validated
against Slurm's syntax), switches logs to `%x_%A_%a` so concurrent tasks
never interleave in one file, and a comment points at
`$SLURM_ARRAY_TASK_ID`, which the entry reads to pick its hyperparameter
row. `watch`/`logs`/`debug` become array-aware
(`logs/<name>_<id>_*.{out,err}`).

### Resource presets (wizard)

Resource selection is explicit, not measured. `lazy107 everything` prints
a numbered preset menu before the per-field editor: `cpu-light` (the CPU
default from `derive_defaults`), `cpu-heavy`, `gpu-light` (the GPU
default), `gpu-heavy`, and `gpu-multi` (2 GPUs). Enter keeps the plan's
current values; a numbered pick wires all four keys at once (values equal
to the current plan are dropped, so nothing is rewritten pointlessly);
`c` falls through to the per-field prompts, each printed with common-value
hints. gpu/cpus must parse as ints (reprompt otherwise); mem/time pass
through as strings. Range and safety checks are deliberately deferred to
`validate_plan`, which runs in the plan print immediately after the edits
are wired.

The design trades measurement for honesty: presets are starting points,
not predictions, and the real measurement already exists in the platform —
`lazy107 debug <job_id>` reads the sacct exit code (`0:9` OOM kill →
raise `mem`; `0:15` timeout → raise `time`) and the log signatures. The
preset → submit → bump-on-evidence loop is the same workflow experienced
users follow by hand; the wizard just makes the first pick cheap.

### Diagnosis (`core/diagnose.py` + `cluster/monitor.py`)

`lazy107 debug <job_id>` explains why a job failed from two evidence
layers (harvested live on the 107 cluster, 2026-09-03 — see
examples/failures/SIGNATURES.md):

1. **Standardized Slurm** — `sacct` exit codes: `1:0` app exit, `0:9`
   SIGKILL (usually OOM → raise `mem`), `0:15` SIGTERM (wall-time or
   cancel → raise `time` / checkpoint).
2. **Log signatures** — regexes over the job's `.err`/`.out` logs, ordered
   most-specific first: `ModuleNotFoundError`/`ImportError`/
   `FileNotFoundError` (parsed names), the timestamped `*** JOB ...
   CANCELLED ... DUE TO TIME LIMIT ***` line, `Detected N oom-kill
   event(s)`, `torch.OutOfMemoryError` (parsed requested/free/total),
   and a generic `XxxError/Exception` catch-all. A log signature always
   beats the exit-code fallback.

`diagnose.py` is pure (no I/O, no subprocess). `cluster/monitor.py`
collects evidence: `sacct_row()` reads `sacct -j <id> -X --parsable2`,
`collect_logs()` globs both log halves (`--error` goes to `.err`, where
tracebacks land) capped at 64 KiB. Degradation: unknown job id warns,
empty logs fall back to the exit code, exit `0:0` without a signature
reports "no failure detected", and unrecognized text prints the last
five log lines for manual reading.

### Manifest (`manifest.py`)

`107.toml` may use flat keys or a `[run]` section. The per-user global
config lives at `~/.config/lazy107/config.toml` (written by `discover`;
invalid or absent files are ignored). `LAZY107_<FIELD>` env vars (e.g.
`LAZY107_GPU=2`) coerce cpus/gpu to int. Merge order: defaults <
inference < global config < manifest < env < flags.

### Discovery (`cluster/discover.py`)

`resolve_association()` parses `sacctmgr show assoc --parsable2` (machine
format: no column truncation) and `scontrol show partition` (regex over the
packed `AllowGroups/AllowAccounts/AllowQos/Default/MaxTime/MaxMemPerNode`
line), then picks the best combo deterministically: default account (+4),
default partition (+2), an explicitly-associated partition (+1); the first
assigned-and-allowed QoS wins. `submission_check()` is the submit-time
guard with the same data, failing open when Slurm data is unavailable and
blocking only on positive evidence of a bad combo or a `time`/`mem` request
exceeding the partition's `MaxTime`/`MaxMemPerNode`. `write_global_config()`
is the only write — to the user's own `~/.config` file.

### Template (`template.py`)

Templates live inside the package (`importlib.resources.files("lazy107")`)
so they survive wheel installs (defect #11). `scaffold()` refuses to touch a
non-empty target, copies every file, and substitutes `{project}`.

### Transfer (`transfer.py`)

One pattern set governs both `.gitignore` (in the template) and `transfer`:
`LARGE_SUFFIXES` (`.pt .pth .ckpt .safetensors .onnx .h5 .hdf5 .npz .npy
.tar .zip .gz .bin .db .sqlite .parquet`) and `LARGE_DIRS` (`data datasets
checkpoints outputs logs models`) — a test asserts the template .gitignore
lists exactly these (defect #10). The checklist excludes its own
`*-data.tar.gz` archive (defect #12) and prints four steps: tar → GUI
upload → extract → sha256.

### Cluster env (`cluster/env.py`)

Conda discovery: PATH first, then `~/miniconda3/bin/conda`. Env existence
via `conda env list --json`. Command sequence is deterministic and
printable:

```bash
module load miniconda/py312  # no-op if conda is already on PATH
conda create -y -n <project> python=3.12
conda env update -n <project> -f environment.yml   # when present
conda run -n <project> pip install -r requirements.txt   # when present
# GPU jobs only: pip overrides, never a conda-channel CPU torch
```

Manifest installs are additive (both `environment.yml` and
`requirements.txt` are honored; `pyproject.toml` only when no
`requirements.txt`). Dependency detection (`core/defaults.py::
scan_imports`) also scans `.py` files (and `.ipynb` code cells) for
third-party imports — stdlib and the project's own modules skipped,
hidden dirs excluded, comma-separated `import a, b` fully captured,
undecodable (non-UTF-8) files ignored — so a project with only code
still gets GPU inference and installed packages. Each import installs
under its own name (`_pip_name` identity fallback), with
`KNOWN_PACKAGES` correcting only the few mismatches
(`sklearn`→`scikit-learn`, `PIL`→`Pillow`, ...). `cmd_env` writes the
scanned packages to `requirements.txt` when neither `requirements.txt`
nor `pyproject.toml` exists. After installing, a GPU job whose
dependencies include the torch family fails closed when torch is not
importable or is a CPU-only build (jax/tf projects skip the torch check).

`env` executes only with `--yes` (dry-run is the read-only preview).
Install commands run under `bash -lc` with a per-command timeout; conda
probes (`env_exists`, `gpu_build_check`) have their own timeout.
Idempotency: `env --yes` skips the create step when the env already
exists. When the plan has no `conda_env`, a successful `env --yes`
wires the env to submissions: `manifest.py::wire_conda_env` writes
`conda_env = "<env>"` into the project's 107.toml — creating a minimal
file when absent, appending without clobbering other keys, replacing an
empty `conda_env = ""` template line, and never overwriting a non-empty
value (a mismatch is reported instead). Dry-run previews the write.
As belt-and-braces, `submit` warns (non-blocking) when an env named
after the project exists but `conda_env` is unset.

Interactive reuse (`cli.py::_prompt_reuse_env`): a bare `env` on a TTY
(with no explicit `--name`) first lists the user's existing envs
(`conda env list`) and offers to reuse one. A pick is validated against
the project's detected dependencies — one `conda run` probe checking
`importlib.util.find_spec` for every dep's import name
(`missing_deps`; `IMPORT_BY_PIP` maps pip names back to import names,
unmappable names are skipped, a failed probe reports everything
missing) — plus, for GPU jobs with torch deps, the existing
`gpu_build_check`. Invalid picks reprompt with the missing-package
list; Enter declines and keeps the `--yes` install flow. A validated
pick is wired with `wire_conda_env(force=True)` (replaces an existing
`conda_env` pin) and the command prints where to edit `conda_env`
afterwards.

## 7. GPU/CUDA integrity (four enforcement points)

"GPU requested must mean CUDA, never a silent CPU fallback." Enforced at
four independent points:

1. **Install-time pinning** (`cluster/env.py::_gpu_overrides`): when a GPU
   framework is detected in a GPU plan, the *last* install commands are pip
   from the default PyPI mirror — `torch`/`torchvision` (default wheel
   bundles CUDA), `jax[cuda12]`, or `tensorflow` (GPU-enabled default).
   Conda channels (CPU torch) can never be the final install step.
2. **Build verification** (`gpu_build_check`): after install, query
   `torch.version.cuda` in the env. Tri-state: `None` = torch absent, `""` =
   CPU-only build, `"12.x"` = CUDA build. `torch.cuda.is_available()` is
   deliberately *not* used — it is always `False` on the login node and
   proves nothing.
3. **Preflight blocking** (`core/validate.py::gpu_preflight_error`): before
   any submit, `gpu>0` with a CPU-only torch aborts with a fix hint
   (`reinstall via lazy107 env --yes`).
4. **Runtime assertion** (`core/render.py`): when `gpu>0`, the rendered
   sbatch runs a CUDA check guarded by `importlib.util.find_spec("torch")`
   (torch-only-if-present; CPU projects are unaffected). A job that starts
   without CUDA fails loudly instead of training on CPU.

## 8. Errors and safety boundaries

### Error handling

- All user-facing failures are caught in `main()` and printed as
  `lazy107: <message>` to stderr with exit code 1.
- `SubmitError` wraps sbatch failures with captured stderr; `PlanError`
  carries the offending field.
- `env --dry-run` works **without conda installed** — printing commands is
  pure output; only execution requires conda.

### Safety boundaries

- **Login node is job management only.** Nothing trains on the login node.
- **`--dry-run` is read-only** — no writes, no subprocess side effects
  beyond printing. Tested for `render` and `submit`.
- **`transfer` prints only** — no rclone, no Pan, no deletions, no
  automated execution (defect #5/#13-#17).
- **Confirmation is honored** — without `--yes` the user must confirm; `n`
  aborts with zero artifacts (defect #3).
- **GPU means CUDA** — see §7; the four points are each covered by tests.
- **No secrets in files** — no tokens or credentials in configs, scripts,
  logs, or notes. Secrets live in the environment only.
- **Validated identifiers** — entry/job_name/conda_env reject whitespace
  and shell metacharacters; sbatch paths are project-relative.

## 9. Configuration interface

```toml
# 107.toml — created by `lazy107 config --init`
partition = "Students"
qos = "qos_stu_default"     # qos_stu_medium / qos_p107-rtx5090 / qos_p107-a100
account = ""                # empty → Slurm default account; no --account line
cpus = 4
mem = "16G"
gpu = 1
nodes = 1                   # >1 → multi-node allocation
ntasks = 1                  # multi-node DDP: must equal nodes
time = "2:00:00"
log_dir = "logs"
conda_env = "demo"          # empty → no conda block in the script
job_name = ""               # empty → entry stem
command = ""                # set → launch verbatim instead of python/torchrun
array = ""                  # set → Slurm job array, e.g. 1-5%2 or 0,2,4
```

Equivalent `[run]` section form is accepted. Precedence (later wins):

```text
tool defaults < dependency inference < ~/.config/lazy107/config.toml
             < 107.toml < LAZY107_* env < CLI flags
```

The per-user global config is written by `lazy107 discover` and holds only
platform-identity values (`account`/`partition`/`qos`); resource knobs stay
per-project.

Environment variables mirror field names: `LAZY107_PARTITION`,
`LAZY107_QOS`, `LAZY107_ACCOUNT`, `LAZY107_CPUS`, `LAZY107_MEM`,
`LAZY107_GPU`, `LAZY107_NODES`, `LAZY107_NTASKS`, `LAZY107_TIME`,
`LAZY107_LOG_DIR`, `LAZY107_CONDA_ENV`, `LAZY107_JOB_NAME`,
`LAZY107_COMMAND`, `LAZY107_ARRAY`, `LAZY107_ENTRY`. Numeric fields coerce to int.

## 10. Platform facts (USTC 107)

Verified live on the login node (2026-09-01, `sacctmgr`/`scontrol`; see also
runbook Step 8):

- Partitions: `P107-RTX5090` (**default**, 15×RTX5090 nodes, 120 GPUs),
  `P107-A100` (11×A100, MaxNodes=2), `GPU-RTX5090`, `GPU-A100`,
  `CPU-6530`, `CPU-8358P`, `Students` (26 nodes).
- Access control: `P107-*` allow only account `competition`
  (`AllowAccounts=competition`) with `qos_p107-rtx5090` / `qos_p107-a100`;
  `Students` allows `stu,stu001,demo_admin,cmet` with the `qos_stu_*` family.
  A competition account on `Students` fails at sbatch time — use `discover`.
- QoS limits: `qos_p107-rtx5090`/`qos_p107-a100`: MaxWall=4d, MaxTRESPU
  cpu=16 gpu=4, MaxJobsPU=4.
- Modules: `miniconda/py312`, `python3.12`, `cuda/12.6` / `cuda/13.0`,
  `apptainer`.
- Mirrors: pip `https://mirrors.ustc.edu.cn/pypi/web/simple`; conda
  channels `anaconda` / `pkgs/main` / `pkgs/free`.
- Conda batch activation: `source "$(conda info --base)/etc/profile.d/conda.sh"`
  then `conda activate` — never `source activate`.
