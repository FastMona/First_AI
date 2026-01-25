"""OOD computation utilities for Mahalanobis-based thresholds.

Consumed by nn_train_cnn/ffn/art and the first_ai CLI. Models must expose
get_features(X) returning embeddings used for distance calculations.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_class_prototypes(
    model,
    train_loader: DataLoader,
    num_classes: int = 10,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """Compute class prototype (mean feature) vectors from training data."""
    model.eval()
    class_features = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device):
                feats = model.get_features(X)

            for i in range(num_classes):
                mask = y == i
                if mask.any():
                    class_features[i].append(feats[mask].cpu())

    class_means: Dict[int, torch.Tensor] = {}
    for i in range(num_classes):
        if class_features[i]:
            all_feats = torch.cat(class_features[i], dim=0)
            class_means[i] = all_feats.mean(dim=0)
            logger.info(f"  Class {i}: {len(all_feats)} samples, prototype computed")

    return class_means


def compute_covariance_matrix(
    model,
    train_loader: DataLoader,
    class_means: Dict[int, torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """Compute diagonal covariance (precision) matrix from training data."""
    model.eval()
    centered = []

    with torch.no_grad():
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device):
                feats = model.get_features(X)

            for i in range(len(y)):
                label = y[i].item()
                if label in class_means:
                    diff = feats[i].cpu() - class_means[label]
                    centered.append(diff)

    if not centered:
        raise RuntimeError("No features collected to compute covariance")

    all_centered = torch.stack(centered, dim=0)
    variance = torch.var(all_centered, dim=0) + 1e-4  # regularize
    precision_diag = 1.0 / variance
    logger.info(f"✓ Diagonal covariance computed (dim={variance.shape[0]})")
    return precision_diag


def compute_mahalanobis_thresholds(
    model,
    val_loader: DataLoader,
    class_means: Dict[int, torch.Tensor],
    precision_diag: torch.Tensor,
    num_classes: int = 10,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute per-class Mahalanobis thresholds on validation data."""
    model.eval()
    class_distances = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device):
                feats = model.get_features(X)

            for i in range(len(y)):
                label = y[i].item()
                if label in class_means:
                    diff = feats[i].cpu() - class_means[label]
                    dist = torch.sqrt(torch.sum(diff ** 2 * precision_diag)).item()
                    class_distances[label].append(dist)

    thresholds_90: Dict[int, float] = {}
    thresholds_95: Dict[int, float] = {}
    thresholds_99: Dict[int, float] = {}
    class_mean: Dict[int, float] = {}
    class_std: Dict[int, float] = {}

    for i in range(num_classes):
        if class_distances[i]:
            arr = np.array(class_distances[i])
            thresholds_90[i] = float(np.percentile(arr, 90))
            thresholds_95[i] = float(np.percentile(arr, 95))
            thresholds_99[i] = float(np.percentile(arr, 99))
            class_mean[i] = float(np.mean(arr))
            class_std[i] = float(np.std(arr))
            logger.info(
                f"  Class {i}: n={len(arr)}, mean={class_mean[i]:.2f} ± {class_std[i]:.2f}, "
                f"90th={thresholds_90[i]:.2f}, 95th={thresholds_95[i]:.2f}, 99th={thresholds_99[i]:.2f}"
            )

    all_distances = [d for values in class_distances.values() for d in values]
    if not all_distances:
        raise RuntimeError("No validation distances to compute thresholds")

    global_threshold_95 = float(np.percentile(all_distances, 95))
    global_mean = float(np.mean(all_distances))

    logger.info(f"Global mean distance: {global_mean:.2f}")
    logger.info(f"Global 95th percentile: {global_threshold_95:.2f}")

    return {
        "class_means": class_means,
        "precision_diag": precision_diag,
        "class_thresholds_90": thresholds_90,
        "class_thresholds_95": thresholds_95,
        "class_thresholds_99": thresholds_99,
        "class_mean_distances": class_mean,
        "class_std_distances": class_std,
        "threshold_95": global_threshold_95,
        "mean_distance": global_mean,
    }
