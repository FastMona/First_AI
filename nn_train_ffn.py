"""Training program for FFN classifier only.

Depends on first_ai.data/build_mnist_dataloaders and first_ai.train for early
stopping. Autoencoder training and OOD parameter computation are handled
separately.
"""

import logging
from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import Config
from nn_model_ffn import FeedforwardClassifier

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

LAST_TRAINING_FOLDER_FILE = Path(".last_training_folder.txt")


def load_last_training_folder(default_folder: Path) -> Path:
    if LAST_TRAINING_FOLDER_FILE.exists():
        value = LAST_TRAINING_FOLDER_FILE.read_text(encoding="utf-8").strip()
        if value:
            return Path(value)
    return default_folder


def normalize_dataset_root(path: Path) -> Path:
    # Accept either dataset root (.../training_data) or raw folder (.../MNIST/raw).
    if (path / "train-images-idx3-ubyte").exists() and (path / "train-labels-idx1-ubyte").exists():
        return path.parent.parent
    return path


def choose_training_folder(default_folder: Path) -> Path:
    last_used = load_last_training_folder(default_folder)
    raw_input = input(
        "Training folder (press Enter to use last one) "
        f"[{last_used}]: "
    ).strip()
    selected = Path(raw_input) if raw_input else last_used
    dataset_root = normalize_dataset_root(selected)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Training folder does not exist: {dataset_root}")

    LAST_TRAINING_FOLDER_FILE.write_text(str(selected), encoding="utf-8")
    return dataset_root


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
    logger.info(f"\n🔧 FFN Architecture:")
    logger.info(f"  • Layer dimensions: [{Config.INPUT_SIZE**2}] → {Config.FFN_HIDDEN_SIZES} → [{Config.FEATURE_DIM}] → [{Config.NUM_CLASSES}]")

    dataset_root = choose_training_folder(Path("training_data"))
    logger.info(f"  • Training data folder: {dataset_root}")

    train_loader, val_loader, test_loader = build_mnist_dataloaders(
        dataset_root=dataset_root,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=getattr(Config, "TRAIN_RATIO", 0.8),
        seed=getattr(Config, "RANDOM_SEED", 42),
    )

    # Initialize classifier
    clf = FeedforwardClassifier(
        input_size=784, hidden_sizes=[512, 256], embedding_size=128, num_classes=10
    ).to(device)
    optimizer = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_with_early_stopping(
        clf,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        num_epochs=epochs,
        patience=3,
        save_path=Config.MODEL_PATH_FFN,
        device=device,
        use_amp=use_amp,
    )

    logger.info("\n" + "=" * 60)
    logger.info("FFN Training Complete")
    logger.info("=" * 60)
    logger.info(f"  - {Config.MODEL_PATH_FFN} (FFN classifier)")
    logger.info("  - Next: Train CCA and compute OOD params (separate options)")


if __name__ == "__main__":
    main()
