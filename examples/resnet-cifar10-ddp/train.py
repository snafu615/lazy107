"""ResNet-18 on CIFAR-10 with DistributedDataParallel (multi-GPU).

Runs under torchrun — lazy107 renders that launch line automatically when
it detects DDP in the project code:

    torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE train.py

Each rank trains on a disjoint shard of the training set (DistributedSampler);
rank 0 evaluates on the full test set and writes outputs/metrics.json, so the
timing recorded there is directly comparable with the single-GPU sibling at
examples/resnet-cifar10/ (same model, same batch size per GPU).

Local check (single GPU):  torchrun --standalone --nproc_per_node=1 train.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from tqdm import tqdm

from data import build_transform
from model import build_model

NUM_CLASSES = 10


def seed_everything(seed: int) -> None:
    """Make runs reproducible across ranks: python, torch, and CUDA."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(data_dir, batch_size, num_workers, rank, world_size):
    """Train loader with a per-rank DistributedSampler; plain test loader."""
    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=build_transform(train=True)
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=build_transform(train=False)
    )
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader, train_sampler


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=f"epoch {epoch:>3} train", unit="batch", leave=False)
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
        bar.set_postfix(loss=f"{running_loss / total:.3f}", acc=f"{correct / total:.3f}")
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet-18 on CIFAR-10 with DDP")
    parser.add_argument("--epochs", type=int, default=2, help="training epochs (default: 2)")
    parser.add_argument("--batch-size", type=int, default=128, help="samples per batch per GPU (default: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate (default: 0.1)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--data-dir", default="data", help="CIFAR-10 root (downloaded if missing)")
    parser.add_argument("--output-dir", default="outputs", help="checkpoints and metrics")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers per rank (default: 0 on Windows, 2 elsewhere)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    seed_everything(args.seed)
    if rank == 0:
        print(
            f"world={world_size} device={device} seed={args.seed} epochs={args.epochs} "
            f"batch={args.batch_size} (per GPU)"
        )

    train_loader, test_loader, train_sampler = make_loaders(
        args.data_dir, args.batch_size, args.num_workers, rank, world_size
    )
    if rank == 0:
        print(f"train={len(train_loader.dataset)} test={len(test_loader.dataset)}")

    model = DistributedDataParallel(build_model(NUM_CLASSES).to(device), device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    per_epoch_s: list[float] = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        per_epoch_s.append(time.perf_counter() - epoch_start)
        if rank == 0:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            lr = scheduler.get_last_lr()[0]
            print(
                f"epoch {epoch:>3} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
                f"| test_loss {test_loss:.4f} test_acc {test_acc:.4f} | lr {lr:.2e} "
                f"| {per_epoch_s[-1]:.1f}s"
            )
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.module.state_dict(), "acc": best_acc},
                    output_dir / "best.pt",
                )

    if rank == 0:
        elapsed = time.perf_counter() - start
        metrics = {
            "best_test_acc": round(best_acc, 4),
            "epochs": args.epochs,
            "device": "cuda",
            "world_size": world_size,
            "elapsed_s": round(elapsed, 1),
            "per_epoch_s": [round(x, 1) for x in per_epoch_s],
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(
            f"done: best_test_acc={best_acc:.4f} in {elapsed / 60:.1f} min on "
            f"{world_size} GPU(s); metrics at {output_dir / 'metrics.json'}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
