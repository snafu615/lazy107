# resnet-cifar10-ddp

Two-GPU variant of `examples/resnet-cifar10/`: same ResNet-18, same CIFAR-10
pipeline, but the entry wraps the model in `DistributedDataParallel`.

`lazy107` detects the DDP code (the `detect_ddp` scanner in
`src/lazy107/core/defaults.py`) and renders the launch line automatically:

```bash
torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE train.py
```

`107.toml` requests `gpu = 2` (plus doubled CPU/memory), so the rendered
sbatch contains `--gres=gpu:2` and the torchrun launch line instead of a
plain `python train.py`. Compare with the single-GPU sibling: `train.py`
here records per-epoch wall time in `outputs/metrics.json` next to
`world_size`, so the 1-GPU vs 2-GPU time-per-epoch numbers are directly
comparable.

Local check (single GPU): `torchrun --standalone --nproc_per_node=1 train.py`
