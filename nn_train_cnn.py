"""Training program for CNN classifier only.

Depends on first_ai.data/build_mnist_dataloaders and first_ai.train for early
stopping. Autoencoder training and OOD parameter computation are handled
separately.
"""

import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import Config
from nn_model_cnn import ImageClassifier

# Configure logging to display output
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Ensure src/ is importable when running as a script
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.append(str(SRC_DIR))

from first_ai.train import train_with_early_stopping  # type: ignore
from first_ai.data import build_mnist_dataloaders  # type: ignore
from first_ai.logging_utils import get_environment_info, log_environment_block  # type: ignore

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main(
    device: str = "auto",
    batch_size: int = 256,
    num_workers: int = 4,
    epochs: int = 10,
):
    device = resolve_device(device)
    use_amp = device.startswith("cuda")

    env_info = get_environment_info()
    log_environment_block(logger, env_info)

    logger.info("\n🚀 Training setup:")
    logger.info(f"  • Training device: {device.upper()}")
    logger.info(f"  • Batch size: {batch_size}")
    logger.info(f"  • Data workers: {num_workers}")
    logger.info(f"  • Mixed precision: {use_amp}")

    train_loader, val_loader, test_loader = build_mnist_dataloaders(
        dataset_root=Path("training_data"),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=getattr(Config, "TRAIN_RATIO", 0.8),
        seed=getattr(Config, "RANDOM_SEED", 42),
    )

    # Initialize classifier
    clf = ImageClassifier().to(device)
    optimizer = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Train classifier with early stopping
    train_with_early_stopping(
        clf,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        num_epochs=epochs,
        patience=3,
        save_path=Config.MODEL_PATH_CNN,
        device=device,
        use_amp=use_amp,
    )

    logger.info("\n" + "=" * 60)
    logger.info("CNN Training Complete")
    logger.info("=" * 60)
    logger.info(f"  - {Config.MODEL_PATH_CNN} (CNN classifier)")
    logger.info("  - Next: Train CCA and compute OOD params (separate options)")


if __name__ == "__main__":
    main()
