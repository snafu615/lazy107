"""CIFAR-10 dataset loading and preprocessing.

Downloads the dataset into *data_dir* on first use (~170 MB), then reuses it.
"""

from __future__ import annotations

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard CIFAR-10 channel statistics.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_transform(train: bool) -> transforms.Compose:
    """Data augmentation for training; normalization only for evaluation."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def load_data(data_dir: str = "data", batch_size: int = 128, num_workers: int | None = None):
    """Return (train_loader, test_loader) for CIFAR-10.

    Args:
        data_dir: Root directory for the CIFAR-10 files (downloaded if missing).
        batch_size: Samples per batch.
        num_workers: DataLoader workers; defaults to 0 on Windows, 2 elsewhere.

    Returns:
        Tuple of training and test DataLoaders.
    """
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=build_transform(train=True)
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=build_transform(train=False)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
