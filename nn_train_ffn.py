"""Training program for FFN classifier and class-conditional autoencoder."""

import logging
from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from autoencoder_model import MNISTAutoencoder
from config import Config
from nn_model_ffn import FeedforwardClassifier

# Ensure src/ is importable when running as a script
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.append(str(SRC_DIR))

from first_ai.train import train_with_early_stopping  # type: ignore
from first_ai.ood import (  # type: ignore
    compute_class_prototypes,
    compute_covariance_matrix,
    compute_mahalanobis_thresholds,
)
from first_ai.ae_train import train_autoencoder, calibrate_reconstruction_threshold  # type: ignore

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_dataloaders(batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    full_train = datasets.MNIST(root="training_data", train=True, download=True, transform=ToTensor())
    test = datasets.MNIST(root="training_data", train=False, download=True, transform=ToTensor())

    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    seed = getattr(Config, "RANDOM_SEED", 42)
    train, validation = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info("Dataset split:")
    logger.info(f"  Training: {len(train)} samples (for model training)")
    logger.info(f"  Validation: {len(validation)} samples (for threshold calibration)")
    logger.info(f"  Test: {len(test)} samples (for final evaluation)")

    common_args = dict(num_workers=num_workers, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **common_args)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, **common_args)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, **common_args)
    return train_loader, val_loader, test_loader


def main(
    device: str = "auto",
    batch_size: int = 256,
    num_workers: int = 4,
    epochs: int = 10,
):
    device = resolve_device(device)
    use_amp = device.startswith("cuda")

    logger.info("\n🚀 GPU Optimization enabled:")
    logger.info(f"  • Batch size: {batch_size}")
    logger.info(f"  • Data workers: {num_workers}")
    logger.info(f"  • Mixed precision: {use_amp}")
    logger.info("")

    train_loader, val_loader, test_loader = build_dataloaders(batch_size, num_workers)

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

    # OOD parameters
    logger.info("\n" + "=" * 60)
    logger.info("Computing Mahalanobis distance parameters for OOD detection")
    logger.info("=" * 60)

    class_means = compute_class_prototypes(clf, train_loader, num_classes=10, device=device)
    precision_diag = compute_covariance_matrix(clf, train_loader, class_means, device=device)
    ood_params = compute_mahalanobis_thresholds(
        clf, val_loader, class_means, precision_diag, num_classes=10, device=device
    )
    ood_params.update({
        "feature_dim": next(iter(class_means.values())).numel() if class_means else 0,
        "model_type": "ffn",
    })
    torch.save(ood_params, Config.OOD_PARAMS_PATH_FFN)
    logger.info(f"\n✓ OOD detection parameters saved to {Config.OOD_PARAMS_PATH_FFN}")

    # Autoencoder training and calibration
    logger.info("\n" + "=" * 60)
    logger.info("Training Class-Conditional Autoencoder")
    logger.info("=" * 60)

    autoencoder = MNISTAutoencoder(latent_dim=64).to(device)
    train_autoencoder(autoencoder, train_loader, test_loader, device=device, epochs=5)

    recon_stats = calibrate_reconstruction_threshold(
        autoencoder,
        clf,
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
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
