"""Data helpers (MNIST, splits, loaders).

Note: This is a scaffold. Actual project scripts may already implement
loading logic; this module centralizes future shared data utilities.
"""

from pathlib import Path
from typing import Tuple

try:
    import torch
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
except Exception:
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    random_split = None  # type: ignore
    datasets = None  # type: ignore
    transforms = None  # type: ignore


def get_mnist_loaders(
    root: Path,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Return training and test MNIST DataLoaders.

    This function assumes torchvision availability.
    """
    if torch is None or datasets is None or transforms is None:
        raise RuntimeError("torch/torchvision not available")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root=str(root), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=str(root), train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
