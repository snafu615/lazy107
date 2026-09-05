# Demo 01 — Install + discover (login node, 0 GPU)

What this proves: lazy107 works with zero configuration. One command reads
your real Slurm association and writes it into a per-user config, which is
the end of the classic `Invalid account or account/partition combination`
sbatch failure.

Capture: `./scripts/capture-demos.sh 01` -> `captures/01/*.txt`.

## Step 0: Install and version check

```bash
pip install -e .
lazy107 --version
```

```text
lazy107 0.1.0
```

A plain Python package, zero runtime dependencies — no uv, no rclone, no
local toolchain. Everything runs on the login node / Web Shell.

## Step 1: The command surface

```bash
lazy107 --help
```

```text
usage: lazy107 [-h] [--version] {init,plan,render,env,submit,watch,status,logs,debug,transfer,discover,config,check,everything} ...

Slurm submission CLI for USTC 107 (cluster side)
```

13 subcommands cover the whole lifecycle: scaffold, plan, render, env,
submit, watch, status, logs, debug, transfer, discover, config, check —
plus the `everything` wizard that chains them. No Slurm parameters to
memorize.

## Step 2: Discover — dry run first

```bash
lazy107 discover --dry-run
```

```text
user=pb24061316
account=competition
partition=P107-RTX5090
qos=qos_p107-rtx5090
# dry-run: not writing ~/.config/lazy107/config.toml
```

These three values come from live `sacctmgr show assoc` / `scontrol show
partition` output — not from a guess. The tool's built-in default
(`Students` / `qos_stu_default`) is *not* authorized for competition
accounts, so resolving the real association is what makes every later demo
submit without errors. The dry run writes nothing (the same read-only
guarantee every `--dry-run` flag carries).

## Step 3: Discover — write it

```bash
lazy107 discover
```

```text
user=pb24061316
account=competition
partition=P107-RTX5090
qos=qos_p107-rtx5090
wrote ~/.config/lazy107/config.toml
```

One-time step. From now on, every project inherits these values through the
global config layer (defaults < inference < global < project `107.toml` <
`LAZY107_*` env < CLI flags).

## Step 4: The effective config

```bash
lazy107 config
```

```text
manifest: 107.toml (absent)
global: ~/.config/lazy107/config.toml (present)
account=competition
partition=P107-RTX5090
qos=qos_p107-rtx5090
cpus=4
mem=16G
gpu=0
...
```

The dashboard shows both config layers and the final merged value of every
field. `partition=P107-RTX5090` comes from the global layer written in
Step 3; nothing else was configured.

## What this proves

- Onboarding is one command: no tokens, no rclone remotes, no manual
  partition tables.
- The account/partition/QoS triple is resolved from live Slurm data
  (`sacctmgr`/`scontrol`), so the "Invalid account" class of submit errors
  is eliminated at the source.
- `--dry-run` is genuinely read-only, verified before anything is written.
