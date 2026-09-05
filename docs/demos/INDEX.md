# lazy107 Demo Series

Six walkthroughs that show what lazy107 does on the real 107 cluster. Each
demo is a numbered step-by-step record (command -> output -> why it matters),
mirroring the format of the competition reference demos. The whole series is
designed for **minimum compute**: about 32 GPU-minutes on `P107-RTX5090`
total, one conda env installed once and reused everywhere, and every
read-only step done on the login node for free.

| # | Demo | Proves | Compute |
|---|---|---|---|
| [01](01-discover.md) | Install + discover | Zero-config onboarding; the fix for "Invalid account or account/partition combination" | login node only |
| [02](02-happy-path.md) | Happy path end-to-end | The whole pipeline, from a code-only project to a trained model on a GPU node | 1 GPU job (~5 min) + one-time env install (~10 min) |
| [03](03-wizard.md) | everything wizard: presets, not probes | Named resource presets wired into the config, per-field custom prompts, and the honest bump-on-evidence loop | 0 GPU (declined submit) |
| [04](04-debug.md) | debug: the tool explains failures | The failure signature library against real 107 log wording | 4 GPU jobs (~8 min total) |
| [05](05-adaptive-render.md) | Adaptive Slurm generation | DDP detection -> torchrun launch, 2-GPU scaling, job arrays, command override | 1 two-GPU job (~3 min) + 1 array (4 tiny tasks) |
| [06](06-guards.md) | Guard rails | Workflow block, honored prompt, config layering, runtime CUDA guard | 1 GPU job (~1 min) + zero-compute blocks |

## How to run

1. Get the repository onto the login node (git clone, or GUI folder upload).
2. `pip install -e .` (or install the wheel).
3. Run the capture script — it executes each demo, waits for submitted jobs
   to reach a terminal state, and saves the verified output:

   ```bash
   ./scripts/capture-demos.sh 01    # ... up to 06, or: all
   ```

   Output lands in `captures/<demo>/*.txt`, each file prefixed with the
   command that produced it. Demos 02-06 submit one job at a time; `all`
   takes about an hour, mostly queue wait.
4. Paste each capture into the matching output block of the demo doc.

## Prerequisites on the cluster

- A Slurm association with `P107-RTX5090` (the competition account) — Demo 01's
  `lazy107 discover` resolves and writes it.
- conda available (`module load miniconda/py312` when not on PATH).
- The `demo` conda env, created once in Demo 02 and reused by every other demo.
- Network to `mirrors.ustc.edu.cn` for the one-time torch install, and to the
  torchvision CIFAR-10 mirror — or pre-stage `data/` via the GUI (Demo 02's
  `lazy107 transfer` shows the upload checklist).

## What to read first

- 01 and 02 are the core story: zero configuration to a trained model.
- 04 and 06 are the differentiators: nobody else explains *why* a job died.
- 03 and 05 show the resource story: presets + honest failure bumps, and
  adaptive Slurm generation.
