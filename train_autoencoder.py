"""Standalone training for the class-conditional autoencoder (CCA).

Trains the shared CCA once and calibrates reconstruction thresholds using
true labels (independent of any classifier). OOD params are computed separately.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

from autoencoder_model import MNISTAutoencoder
from config import Config

# Configure logging to display output
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Ensure src/ is importable when running as a script
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.append(str(SRC_DIR))

from first_ai.ae_train import train_autoencoder  # type: ignore
from first_ai.data import build_mnist_dataloaders  # type: ignore
from first_ai.logging_utils import get_environment_info, log_environment_block  # type: ignore

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def calibrate_reconstruction_threshold_labels(
    autoencoder: MNISTAutoencoder,
    val_loader,
    device: str = "cuda",
    percentiles=(95, 99),
):
    """Calibrate reconstruction thresholds using ground-truth labels."""
    autoencoder.eval()
    errors = []
    use_amp = device.startswith("cuda")

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast(device):
                    batch_errors = autoencoder.reconstruction_error(X, y)
            else:
                batch_errors = autoencoder.reconstruction_error(X, y)
            errors.extend(batch_errors.cpu().tolist())

    if not errors:
        raise RuntimeError("No reconstruction errors computed for calibration")

    errors_np = np.array(errors)
    threshold_low = float(np.percentile(errors_np, percentiles[0]))
    threshold_high = float(np.percentile(errors_np, percentiles[1]))

    return {
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
        "mean_error": float(np.mean(errors_np)),
        "std_error": float(np.std(errors_np)),
    }


def main(
    device: str = "auto",
    batch_size: int = 256,
    num_workers: int = 4,
    epochs: int = Config.AE_EPOCHS,
):
    device = resolve_device(device)
    use_amp = device.startswith("cuda")

    env_info = get_environment_info()
    log_environment_block(logger, env_info)

    logger.info("\n🚀 Autoencoder training setup:")
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

    logger.info("\n" + "=" * 60)
    logger.info("Training Class-Conditional Autoencoder")
    logger.info("=" * 60)

    autoencoder = MNISTAutoencoder(
        latent_dim=Config.LATENT_DIM,
        embedding_dim=Config.EMBEDDING_DIM,
    ).to(device)

    train_autoencoder(autoencoder, train_loader, test_loader, device=device, epochs=epochs)

    recon_stats = calibrate_reconstruction_threshold_labels(
        autoencoder,
        val_loader,
        device=device,
        percentiles=(95, 99),
    )

    torch.save(
        {
            "model_state": autoencoder.state_dict(),
            "threshold_95": recon_stats["threshold_low"],
            "threshold_99": recon_stats["threshold_high"],
            "mean_error": recon_stats["mean_error"],
            "std_error": recon_stats["std_error"],
        },
        Config.AUTOENCODER_PATH,
    )

    logger.info(f"\n✓ Autoencoder saved to {Config.AUTOENCODER_PATH}")
    logger.info(
        f"  - Reconstruction threshold (95%): {recon_stats['threshold_low']:.6f}"
    )
    logger.info("  - Use as Stage 1 gate for OOD detection")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
