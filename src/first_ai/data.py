"""Data helpers (MNIST, splits, loaders).

Used by nn_train_cnn/ffn/art and the first_ai CLI to produce shared
train/val/test loaders. Keeps defaults consistent with legacy scripts
(ToTensor only, no normalization) while allowing configurable split ratios
and seeds.
"""

import logging
from pathlib import Path
from typing import Tuple, Union

try:
    import torch
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets
    from torchvision.transforms import ToTensor
except Exception:
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    random_split = None  # type: ignore
    datasets = None  # type: ignore
    ToTensor = None  # type: ignore


logger = logging.getLogger(__name__)


def build_mnist_dataloaders(
    dataset_root: Union[Path, str] = Path("training_data"),
    train_batch_size: int = 256,
    eval_batch_size: int = 256,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test loaders for MNIST with a configurable split."""
    if torch is None or datasets is None or ToTensor is None or random_split is None:
        raise RuntimeError("torch/torchvision not available")

    dataset_root = Path(dataset_root)
    tfm = ToTensor()
    full_train = datasets.MNIST(root=str(dataset_root), train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=str(dataset_root), train=False, download=True, transform=tfm)

    train_size = int(train_ratio * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info("Dataset split:")
    logger.info(f"  Training: {len(train_set)} samples (for model training)")
    logger.info(f"  Validation: {len(val_set)} samples (for threshold calibration)")
    logger.info(f"  Test: {len(test)} samples (for final evaluation)")

    common_eval_args = dict(num_workers=num_workers, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, **common_eval_args)
    test_loader = DataLoader(test, batch_size=eval_batch_size, shuffle=False, **common_eval_args)
    return train_loader, val_loader, test_loader
