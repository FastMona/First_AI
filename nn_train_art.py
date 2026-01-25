"""Training program for Fuzzy ART and class-conditional autoencoder."""

import logging
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn, save
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from autoencoder_model import MNISTAutoencoder
from config import Config
from nn_model_art import FuzzyARTClassifier

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.append(str(SRC_DIR))

from first_ai.ood import compute_class_prototypes, compute_covariance_matrix, compute_mahalanobis_thresholds  # type: ignore
from first_ai.ae_train import train_autoencoder  # type: ignore

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_dataloaders(train_batch_size: int, eval_batch_size: int, num_workers: int):
    full_train = datasets.MNIST(root="training_data", train=True, download=True, transform=ToTensor())
    test = datasets.MNIST(root="training_data", train=False, download=True, transform=ToTensor())

    train_size = int(Config.TRAIN_RATIO * len(full_train))
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

    train_loader = DataLoader(
        train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        validation,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader, test_loader


def train_art(
    art,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    num_passes: int = 3,
) -> None:
    art.train()
    total_batches = len(train_loader)

    for pass_num in range(num_passes):
        logger.info(f"\n{'='*80}")
        logger.info(f"  PASS {pass_num + 1}/{num_passes} - Processing {len(train_loader.dataset)} training samples")
        logger.info(f"{'='*80}")

        total_samples = 0
        pass_start_time = time.time()
        batch_times = []

        for batch_idx, (X, y) in enumerate(train_loader):
            batch_start = time.time()
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            for i in range(X.size(0)):
                art.train_pattern(X[i].view(-1), y[i])
                total_samples += 1

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if (batch_idx + 1) % 50 == 0:
                avg_batch_time = np.mean(batch_times[-50:])
                samples_per_sec = (50 * train_loader.batch_size) / sum(batch_times[-50:])
                progress_pct = (batch_idx + 1) / total_batches * 100
                eta_batches = total_batches - (batch_idx + 1)
                eta_seconds = eta_batches * avg_batch_time
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                current_time = time.strftime("%H:%M")

                logger.info(
                    f"{progress_pct:.1f}% complete | Batch {batch_idx + 1}/{total_batches} | "
                    f"Samples Processed: {total_samples} | Speed: {samples_per_sec:.1f} samp/sec | "
                    f"Time remaining: {eta_min:02d}:{eta_sec:02d} | {current_time}"
                )

        pass_time = time.time() - pass_start_time
        pass_min = int(pass_time // 60)
        pass_sec = int(pass_time % 60)

        logger.info(f"\n✓ Pass {pass_num + 1} complete in {pass_min}m {pass_sec}s")
        logger.info(f"  Total samples processed: {total_samples}")
        logger.info(f"  Categories committed: {art.num_committed}/{Config.ART_MAX_CATEGORIES}")
        logger.info(f"  Average speed: {total_samples/pass_time:.1f} samples/s")

        # Evaluate after each pass
        art.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                X_flat = X.view(X.size(0), -1)
                with torch.amp.autocast(device):
                    logits = art.predict(X_flat)
                    _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = 100 * correct / total
        logger.info(f"  Test Accuracy after pass {pass_num + 1}: {accuracy:.2f}% ({correct}/{total} correct)")
        art.train()


def calibrate_reconstruction_threshold_art(autoencoder, art, val_loader, device: str = "cuda"):
    autoencoder.eval()
    art.eval()
    recon_errors = []

    with torch.no_grad():
        for X, _ in val_loader:
            X = X.to(device, non_blocking=True)
            with torch.amp.autocast(device):
                X_flat = X.view(X.size(0), -1)
                preds = torch.argmax(art.predict(X_flat), dim=1)
                errors = autoencoder.reconstruction_error(X, preds)
            recon_errors.extend(errors.cpu().tolist())

    recon_errors = np.array(recon_errors)
    recon_threshold_95 = float(np.percentile(recon_errors, 95))
    recon_threshold_99 = float(np.percentile(recon_errors, 99))
    recon_mean = float(np.mean(recon_errors))
    recon_std = float(np.std(recon_errors))

    logger.info("\nReconstruction error statistics on VALIDATION data:")
    logger.info(f"  Samples: {len(recon_errors)}")
    logger.info(f"  Mean: {recon_mean:.6f}")
    logger.info(f"  Std: {recon_std:.6f}")
    logger.info(f"  95th percentile: {recon_threshold_95:.6f}")
    logger.info(f"  99th percentile: {recon_threshold_99:.6f}")

    return recon_threshold_95, recon_threshold_99, recon_mean, recon_std


def main(
    device: str = "auto",
    train_batch_size: int = 64,
    eval_batch_size: int = 256,
    num_workers: int = 4,
    passes: int = 3,
):
    device = resolve_device(device)

    logger.info("\n" + "=" * 80)
    logger.info("  Training Fuzzy Adaptive Resonance Theory (ART) Network".center(80))
    logger.info("=" * 80)
    logger.info(f"  Device: {device}")
    logger.info(f"  Train batch size: {train_batch_size}")
    logger.info(f"  Eval batch size: {eval_batch_size}")
    logger.info(f"  Passes: {passes}")

    train_loader, val_loader, test_loader = build_dataloaders(train_batch_size, eval_batch_size, num_workers)

    art = FuzzyARTClassifier(
        input_dim=Config.INPUT_SIZE * Config.INPUT_SIZE,
        max_categories=Config.ART_MAX_CATEGORIES,
        vigilance=Config.ART_VIGILANCE,
        learning_rate=Config.ART_LEARNING_RATE,
        choice_alpha=Config.ART_CHOICE_ALPHA,
    ).to(device)

    # ART training
    train_art(art, train_loader, test_loader, device=device, num_passes=passes)

    logger.info(f"\n✓ Saving ART model to {Config.MODEL_PATH_ART}")
    with open(Config.MODEL_PATH_ART, "wb") as f:
        save(art.state_dict(), f)

    # OOD parameters
    logger.info("\n" + "=" * 60)
    logger.info("Computing OOD Detection Parameters for ART Model")
    logger.info("=" * 60)

    class_means = compute_class_prototypes(art, train_loader, num_classes=Config.NUM_CLASSES, device=device)
    precision_diag = compute_covariance_matrix(art, train_loader, class_means, device=device)
    ood_params = compute_mahalanobis_thresholds(
        art,
        val_loader,
        class_means,
        precision_diag,
        num_classes=Config.NUM_CLASSES,
        device=device,
    )
    ood_params.update({
        "feature_dim": art.coded_dim,
        "model_type": "art",
    })
    torch.save(ood_params, Config.OOD_PARAMS_PATH)
    logger.info(f"\n✓ OOD detection parameters saved to {Config.OOD_PARAMS_PATH}")

    # Autoencoder training and calibration
    logger.info("\n" + "=" * 60)
    logger.info("Training Class-Conditional Autoencoder")
    logger.info("=" * 60)

    autoencoder = MNISTAutoencoder(
        latent_dim=Config.LATENT_DIM,
        embedding_dim=Config.EMBEDDING_DIM,
    ).to(device)

    train_autoencoder(autoencoder, train_loader, test_loader, device=device, epochs=Config.AE_EPOCHS)

    recon_threshold_95, recon_threshold_99, recon_mean, recon_std = calibrate_reconstruction_threshold_art(
        autoencoder, art, val_loader, device=device
    )

    torch.save(
        {
            "model_state": autoencoder.state_dict(),
            "threshold_95": recon_threshold_95,
            "threshold_99": recon_threshold_99,
            "mean_error": recon_mean,
            "std_error": recon_std,
        },
        Config.AUTOENCODER_PATH,
    )

    logger.info(f"\n✓ Autoencoder saved to {Config.AUTOENCODER_PATH}")
    logger.info(f"  - Reconstruction threshold (95%): {recon_threshold_95:.6f}")
    logger.info("=" * 60)

    logger.info("\n" + "=" * 80)
    logger.info("  ART Training Complete!".center(80))
    logger.info("=" * 80)
    logger.info(f"  - {Config.MODEL_PATH_ART} (ART classifier)")
    logger.info(f"  - {Config.AUTOENCODER_PATH} (Autoencoder)")
    logger.info(f"  - {Config.OOD_PARAMS_PATH} (OOD detection parameters)")


if __name__ == "__main__":
    main()
