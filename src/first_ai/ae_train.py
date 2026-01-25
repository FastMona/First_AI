"""Autoencoder training utilities for class-conditional reconstruction.

Used by nn_train_cnn/ffn/art and the first_ai CLI. Requires autoencoders
to support reconstruction_error(pred_classes) for threshold calibration.
"""

import logging
from typing import Dict, Any, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_autoencoder(
    autoencoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 5,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """Train class-conditional autoencoder and return loss history."""
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(device) if device.startswith("cuda") else None

    history = {"train_loss": [], "val_loss": []}
    autoencoder.to(device)

    for epoch in range(epochs):
        autoencoder.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device):
                    recon = autoencoder(X, y)
                    loss = loss_fn(recon, X)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon = autoencoder(X, y)
                loss = loss_fn(recon, X)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation reconstruction loss
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device):
                    recon = autoencoder(X, y)
                    loss = loss_fn(recon, X)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger.info(
            f"AE Epoch {epoch}: Train Recon Loss = {train_loss:.6f}, Val Recon Loss = {val_loss:.6f}"
        )

    return history


def calibrate_reconstruction_threshold(
    autoencoder: nn.Module,
    classifier: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    percentiles: Tuple[int, int] = (95, 99),
) -> Dict[str, float]:
    """Calibrate reconstruction-error thresholds using classifier predictions."""
    autoencoder.eval()
    classifier.eval()
    errors = []

    with torch.no_grad():
        for X, _ in val_loader:
            X = X.to(device, non_blocking=True)
            with torch.amp.autocast(device):
                preds = torch.argmax(classifier(X), dim=1)
                batch_errors = autoencoder.reconstruction_error(X, preds)
            errors.extend(batch_errors.cpu().tolist())

    errors_arr = np.array(errors)
    p_low, p_high = percentiles
    threshold_low = float(np.percentile(errors_arr, p_low))
    threshold_high = float(np.percentile(errors_arr, p_high))
    mean_err = float(np.mean(errors_arr))
    std_err = float(np.std(errors_arr))

    logger.info("\nReconstruction error statistics (validation):")
    logger.info(f"  Samples: {len(errors_arr)}")
    logger.info(f"  Mean: {mean_err:.6f}")
    logger.info(f"  Std: {std_err:.6f}")
    logger.info(f"  {p_low}th percentile: {threshold_low:.6f}")
    logger.info(f"  {p_high}th percentile: {threshold_high:.6f}")

    return {
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
        "mean_error": mean_err,
        "std_error": std_err,
        "percentiles": percentiles,
    }
