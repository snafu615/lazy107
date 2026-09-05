"""ResNet-18 training on CIFAR-10.

Pure Python entry point: runs on any machine with torch + torchvision
installed. The device is auto-detected (CUDA when available, CPU otherwise),
so the same code runs locally and as a Slurm job submitted by lazy107.

Run from the project root:

    python train.py --epochs 5
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import load_data
from model import build_model

NUM_CLASSES = 10


def seed_everything(seed: int) -> None:
    """Make runs reproducible: python, torch, and (when available) CUDA."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


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
    parser = argparse.ArgumentParser(description="Train a ResNet-18 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=5, help="training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=128, help="samples per batch (default: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate (default: 0.1)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--data-dir", default="data", help="CIFAR-10 root (downloaded if missing)")
    parser.add_argument("--output-dir", default="outputs", help="checkpoints and metrics")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: 0 on Windows, 2 elsewhere)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = select_device(args.device)
    start = time.perf_counter()
    print(
        f"device={device.type} seed={args.seed} epochs={args.epochs} "
        f"batch={args.batch_size} data_dir={args.data_dir}"
    )

    train_loader, test_loader = load_data(args.data_dir, args.batch_size, args.num_workers)
    print(f"train={len(train_loader.dataset)} test={len(test_loader.dataset)}")

    model = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(
            f"epoch {epoch:>3} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
            f"| test_loss {test_loss:.4f} test_acc {test_acc:.4f} | lr {lr:.2e}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "acc": best_acc},
                checkpoint_path,
            )

    elapsed = time.perf_counter() - start
    metrics = {
        "best_test_acc": round(best_acc, 4),
        "epochs": args.epochs,
        "device": device.type,
        "elapsed_s": round(elapsed, 1),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"done: best_test_acc={best_acc:.4f} in {elapsed / 60:.1f} min on {device.type}; "
        f"checkpoint at {checkpoint_path}"
    )


if __name__ == "__main__":
    main()
