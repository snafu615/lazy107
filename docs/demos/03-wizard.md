# Demo 03 — everything wizard: presets, not probes (0 GPU)

What this proves: the guided `everything` pipeline replaces resource
measurement with honest presets — one numbered pick wires `gpu`/`cpus`/
`mem`/`time` into `107.toml` at once, `c` opens the per-field editor with
common-value hints, and the final submit confirm is the last gate before
anything is queued. Every step here runs on the login node; the submit is
declined, so this demo costs **zero GPU minutes**.

Compute: none — prompts and read-only prints only.

Capture: `./scripts/capture-demos.sh 03` -> `captures/03/*.txt` (drives the
prompts through a pseudo-TTY, so util-linux `script` is required).

## Step 0: The project, already wired from Demo 02

`~/demos/demo` (resnet-cifar10) still has `conda_env = "demo"` from Demo
02's install, and its entry is unpinned — so the wizard opens with the
entry picker, then skips env setup entirely.

## Step 1: Pick a preset

```bash
printf '1\n4\ny\nn\n' | script -qec 'lazy107 everything' /dev/null
```

(Answers: `1` = entry train.py, `4` = the gpu-heavy preset, `y` = preview
the sbatch script, `n` = decline the submit.)

```text
detected entry files:
  1) train.py   <- default
  2) data.py
  3) model.py
choose entry [1-3, Enter=train.py]: 1
pinned entry = 'train.py' in 107.toml
note: conda_env = 'demo' already wired; skipping env setup
slurm resources (Enter keeps the current values):
  1) cpu-light  gpu=0 cpus=2 mem=4G time=1:00:00
  2) cpu-heavy  gpu=0 cpus=8 mem=32G time=8:00:00
  3) gpu-light  gpu=1 cpus=4 mem=16G time=2:00:00
  4) gpu-heavy  gpu=1 cpus=8 mem=64G time=12:00:00
  5) gpu-multi  gpu=2 cpus=16 mem=64G time=24:00:00
pick a preset [1-5], c to customize each field: 4
wired cpus=8, mem='64G', time='12:00:00' into 107.toml
entry=train.py
account=competition
partition=P107-RTX5090
qos=qos_p107-rtx5090
cpus=8
mem=64G
gpu=1
time=12:00:00
...
preview the sbatch script? [y/N] y
#SBATCH --job-name=train
...
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
...
submit train on P107-RTX5090 (gpu=1)? [y/N] n
lazy107: aborted
```

Three things happened. The entry picker pinned `train.py`. The preset menu
replaced the guesswork: `gpu-heavy` wired `cpus`/`mem`/`time` in one answer
(`gpu=1` matched the plan already, so it was dropped — nothing is rewritten
pointlessly). And the declined submit confirms that the wizard's final gate
is a real prompt: nothing was queued, exit 1, zero compute.

## Step 2: The manifest shows exactly what was wired

```bash
cat 107.toml
```

```text
entry = "train.py"
cpus = 8
mem = "64G"
time = "12:00:00"
```

## Step 3: The custom path — `c` opens the per-field editor

```bash
printf 'c\n\n\n\n\nn\nn\n' | script -qec 'lazy107 everything' /dev/null
```

(Answers: `c` = customize, Enter on every field = keep the current value,
`n` = no preview, `n` = decline the submit.)

```text
note: conda_env = 'demo' already wired; skipping env setup
slurm resources (Enter keeps the current values):
  1) cpu-light  gpu=0 cpus=2 mem=4G time=1:00:00
  ...
pick a preset [1-5], c to customize each field: c
  number of GPUs (common: 0, 1, 2) [1]
  CPUs per task (common: 2, 4, 8) [8]
  memory (e.g. 16G; common: 4G, 16G, 32G) [64G]
  time limit (e.g. 1:00:00; common: 1:00:00, 2:00:00) [12:00:00]
...
submit train on P107-RTX5090 (gpu=1)? [y/N] n
lazy107: aborted
```

The entry picker did not reappear (it is pinned now), and every field
prompt prints the current value plus a common-value hint — Enter keeps,
typing replaces. Nothing was wired because nothing changed.

## Step 4: plan re-checks the result

```bash
lazy107 plan
```

```text
entry=train.py
...
cpus=8
mem=64G
gpu=1
time=12:00:00
...
```

`plan` is the ground truth after any edit — the wizard defers all range and
safety checks to it, so a bad value surfaces here with a readable message
instead of being buried in a submit failure.

## What this proves

- Resource selection is explicit: named presets (mirroring the tool's CPU
  and GPU defaults), one pick wires all four keys, `c` keeps full control.
- The wizard pins the entry and writes only what changed; `plan` validates.
- The submit confirm is honored — the whole demo ran with zero queued jobs.
- The replacement for measurement is the honest loop: start from a preset,
  submit, and bump on evidence — Demo 04's `debug` turns an OOM kill
  (`0:9`) into "raise `mem`" and a timeout (`0:15`) into "raise `time`".
