# Demo 05 — Adaptive Slurm generation (2-GPU DDP + job array + override)

What this proves: the sbatch generator adapts to the project — DDP code
gets a torchrun multi-GPU launch, sweep projects get job arrays with
per-task logs, and an explicit `command` overrides everything. Resource
requests follow: 2 GPUs means doubled CPU/memory in the same config.

Compute: one 2-GPU DDP job (~3 min, 2 epochs) + one 4-task array (~1 min of
tiny tasks) + zero-compute renders. All reusing the `demo` env.

Capture: `./scripts/capture-demos.sh 05` -> `captures/05/*.txt`.

## Part A — DDP detection and a real two-GPU run

### Step 0: The project

`examples/resnet-cifar10-ddp/` is the Demo 02 model wrapped in
`DistributedDataParallel` (`train.py` contains `init_process_group(...)` and
`DistributedDataParallel(...)` — real DDP code, not a comment). Its
`107.toml` asks for `gpu = 2` with doubled CPU/memory:

```bash
cd ~/demos/ddp
lazy107 plan
```

```text
entry=train.py
partition=P107-RTX5090
qos=qos_p107-rtx5090
gpu=2
cpus=8
mem=32G
time=0:20:00
...
```

### Step 1: The rendered script adapts

```bash
lazy107 render --dry-run
```

Key lines:

```bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
...
torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE train.py
```

A static scanner (`detect_ddp` in `core/defaults.py`, anchored to import
and call forms so comments can never trigger it) switched the launch line
from `python train.py` to torchrun with one process per allocated GPU.
Nothing was configured by hand beyond `gpu = 2`.

### Step 2: Run it and read the scaling number

```bash
lazy107 submit --yes
# job COMPLETED after ~3 min
cat outputs/metrics.json
```

```text
{
  "best_test_acc": 0.7xxx,
  "epochs": 2,
  "device": "cuda",
  "world_size": 2,
  "elapsed_s": 1xx.x,
  "per_epoch_s": [xx.x, xx.x]
}
```

Compare with Demo 02's single-GPU run (same model, same 128 images per
batch per GPU): divide its `elapsed_s` by its 5 epochs to get its
time-per-epoch. Two GPUs process twice the images per step, so the
time-per-epoch drops accordingly — the demo captures both `metrics.json`
files side by side for the comparison. Also note the CUDA runtime guard
still ran inside the job (it passed — 2 real GPUs), and the log shows both
ranks' progress bars.

## Part B — Job arrays: a hyperparameter sweep in one command

### Step 3: The project

`examples/sweep/` is a 20-line entry that reads `$SLURM_ARRAY_TASK_ID` and
picks one row of a 4-row grid. Its `107.toml` sets `array = "1-4%2"`
(four tasks, two concurrent).

```bash
cd ~/demos/sweep
lazy107 render --dry-run
```

```bash
#SBATCH --array=1-4%2
#SBATCH --output=logs/sweep_%x_%A_%a.out
#SBATCH --error=logs/sweep_%x_%A_%a.err
...
# array task index available as $SLURM_ARRAY_TASK_ID
python train.py
```

Two details matter: the array spec is validated against Slurm's syntax
before rendering, and the log pattern switches to `%A_%a` (master job id +
task index) so concurrent tasks never interleave in one file.

### Step 4: Run it — four tasks, two at a time

```bash
lazy107 submit --yes
squeue -u $USER     # shows the master job; tasks run under it
ls logs
```

```text
sweep_53086_1.err  sweep_53086_1.out  sweep_53086_2.err  sweep_53086_2.out
sweep_53086_3.err  sweep_53086_3.out  sweep_53086_4.err  sweep_53086_4.out
```

Each task wrote its own pair of logs:

```bash
cat logs/sweep_53086_1.out
```

```text
task 1/4: lr=0.001 steps=3
  step 1/3: simulated_loss=0.5000
  step 2/3: simulated_loss=0.2500
  step 3/3: simulated_loss=0.1250
task 1 done: final_loss=0.1250
```

## Part C — `command` override: full user control

### Step 5: An explicit command wins, verbatim

```bash
cd ~/demos/ddp
# temporarily add to 107.toml:
#   command = "python train.py --epochs 1 --batch-size 64"
lazy107 render --dry-run
```

The launch line is replaced verbatim — even the DDP torchrun line:

```bash
python train.py --epochs 1 --batch-size 64
```

(Dry-run only: nothing was submitted, and `107.toml` is restored
afterwards.)

## What this proves

- Slurm generation adapts to the code: DDP signals -> torchrun +
  `--gres=gpu:N`; array spec -> `--array` + per-task logs; `command` ->
  verbatim replacement (highest precedence).
- The DDP detector is anchored to high-confidence forms so false positives
  (comments, `DataParallel`) can never launch the entry N times.
- Multi-GPU is demonstrated end-to-end on real hardware with a measured
  scaling comparison against the single-GPU run of Demo 02.
