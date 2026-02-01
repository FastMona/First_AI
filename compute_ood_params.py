"""Standalone script to compute OOD (Out-of-Distribution) parameters for all trained models.

This generates Mahalanobis distance parameters for Stage 2 OOD detection:
- Class prototypes (mean feature vectors per class)
- Precision matrix (inverse covariance)
- Per-class distance thresholds (90th, 95th, 99th percentiles)

Must be run after training classifiers (FFN/CNN/ART) and before OOD detection.
"""

import logging
import sys
import torch
from pathlib import Path

# Add src directory to path for first_ai imports
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from config import Config
from nn_model_cnn import ImageClassifier as CNNModel
from nn_model_ffn import FeedforwardClassifier as FFNModel
from nn_model_art import FuzzyARTClassifier as ARTModel
from first_ai.data import build_mnist_dataloaders
from first_ai.ood import (
    compute_class_prototypes,
    compute_covariance_matrix,
    compute_mahalanobis_thresholds
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title.center(76)}  ")
    print("=" * 80 + "\n")


def compute_for_cnn():
    """Compute OOD parameters for CNN model."""
    model_path = Config.MODEL_PATH_CNN
    ood_path = Config.OOD_PARAMS_PATH_CNN
    
    if not model_path.exists():
        logger.warning(f"⚠ CNN model not found at {model_path} - skipping")
        return False
    
    logger.info("🔧 Loading CNN model...")
    device = Config.DEVICE
    model = CNNModel().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    logger.info("📊 Loading MNIST dataset...")
    train_loader, val_loader, _ = build_mnist_dataloaders(
        dataset_root='training_data',
        train_batch_size=256,
        eval_batch_size=256,
        num_workers=4
    )
    
    logger.info("🧮 Computing class prototypes...")
    class_means = compute_class_prototypes(
        model, train_loader, num_classes=10, device=device.type
    )
    
    logger.info("🧮 Computing covariance matrix...")
    precision_diag = compute_covariance_matrix(
        model, train_loader, class_means, device=device.type
    )
    
    logger.info("🧮 Computing Mahalanobis thresholds on validation set...")
    ood_params = compute_mahalanobis_thresholds(
        model, val_loader, class_means, precision_diag, num_classes=10, device=device.type
    )
    
    # Add metadata
    ood_params['feature_dim'] = 128
    ood_params['model_type'] = 'cnn'
    
    logger.info(f"💾 Saving OOD parameters to {ood_path}...")
    torch.save(ood_params, ood_path)
    logger.info(f"✓ CNN OOD parameters saved successfully\n")
    return True


def compute_for_ffn():
    """Compute OOD parameters for FFN model."""
    model_path = Config.MODEL_PATH_FFN
    ood_path = Config.OOD_PARAMS_PATH_FFN
    
    if not model_path.exists():
        logger.warning(f"⚠ FFN model not found at {model_path} - skipping")
        return False
    
    logger.info("🔧 Loading FFN model...")
    device = Config.DEVICE
    model = FFNModel(
        input_size=784,
        hidden_sizes=Config.FFN_HIDDEN_SIZES,
        embedding_size=128,
        num_classes=10
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    logger.info("📊 Loading MNIST dataset...")
    train_loader, val_loader, _ = build_mnist_dataloaders(
        dataset_root='training_data',
        train_batch_size=256,
        eval_batch_size=256,
        num_workers=4
    )
    
    logger.info("🧮 Computing class prototypes...")
    class_means = compute_class_prototypes(
        model, train_loader, num_classes=10, device=device.type
    )
    
    logger.info("🧮 Computing covariance matrix...")
    precision_diag = compute_covariance_matrix(
        model, train_loader, class_means, device=device.type
    )
    
    logger.info("🧮 Computing Mahalanobis thresholds on validation set...")
    ood_params = compute_mahalanobis_thresholds(
        model, val_loader, class_means, precision_diag, num_classes=10, device=device.type
    )
    
    # Add metadata
    ood_params['feature_dim'] = 128
    ood_params['model_type'] = 'ffn'
    
    logger.info(f"💾 Saving OOD parameters to {ood_path}...")
    torch.save(ood_params, ood_path)
    logger.info(f"✓ FFN OOD parameters saved successfully\n")
    return True


def compute_for_art():
    """Compute OOD parameters for ART model."""
    model_path = Config.MODEL_PATH_ART
    ood_path = Config.OOD_PARAMS_PATH_ART
    
    if not model_path.exists():
        logger.warning(f"⚠ ART model not found at {model_path} - skipping")
        return False
    
    logger.info("🔧 Loading ART model...")
    # ART uses CPU for sequential processing
    device = torch.device('cpu')
    model = ARTModel(
        input_dim=784,
        max_categories=Config.ART_MAX_CATEGORIES,
        vigilance=Config.ART_VIGILANCE,
        learning_rate=Config.ART_LEARNING_RATE,
        choice_alpha=Config.ART_CHOICE_ALPHA
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    logger.info("📊 Loading MNIST dataset...")
    train_loader, val_loader, _ = build_mnist_dataloaders(
        dataset_root='training_data',
        train_batch_size=256,
        eval_batch_size=256,
        num_workers=4
    )
    
    logger.info("🧮 Computing class prototypes...")
    class_means = compute_class_prototypes(
        model, train_loader, num_classes=10, device='cpu'
    )
    
    logger.info("🧮 Computing covariance matrix...")
    precision_diag = compute_covariance_matrix(
        model, train_loader, class_means, device='cpu'
    )
    
    logger.info("🧮 Computing Mahalanobis thresholds on validation set...")
    ood_params = compute_mahalanobis_thresholds(
        model, val_loader, class_means, precision_diag, num_classes=10, device='cpu'
    )
    
    # Add metadata
    ood_params['feature_dim'] = 1568  # ART uses 784*2 complement coded features
    ood_params['model_type'] = 'art'
    
    logger.info(f"💾 Saving OOD parameters to {ood_path}...")
    torch.save(ood_params, ood_path)
    logger.info(f"✓ ART OOD parameters saved successfully\n")
    return True


def main():
    """Compute OOD parameters for all trained models."""
    print_header("COMPUTE OOD PARAMETERS FOR ALL MODELS")
    
    logger.info("This will compute Mahalanobis distance parameters for Stage 2 OOD detection.")
    logger.info("Ensure you have trained at least one classifier (FFN/CNN/ART) before running.\n")
    
    results = {
        'FFN': False,
        'CNN': False,
        'ART': False
    }
    
    # Compute for each model
    print_header("FFN MODEL")
    results['FFN'] = compute_for_ffn()
    
    print_header("CNN MODEL")
    results['CNN'] = compute_for_cnn()
    
    print_header("ART MODEL")
    results['ART'] = compute_for_art()
    
    # Summary
    print_header("SUMMARY")
    success_count = sum(results.values())
    
    if success_count == 0:
        logger.error("❌ No OOD parameters computed. Train at least one model first.")
        logger.info("\nNext steps:")
        logger.info("  1. Train a classifier: Run dashboard option 1 (Train a Model)")
        logger.info("  2. Then re-run this script (option 2)")
    else:
        logger.info(f"✓ Successfully computed OOD parameters for {success_count} model(s):")
        for model_name, success in results.items():
            status = "✓" if success else "○"
            logger.info(f"  {status} {model_name}")
        
        logger.info("\nNext steps:")
        logger.info("  - To use two-stage OOD detection, also train CCA (option 1→4)")
        logger.info("  - Test detection: Run dashboard option 4 or 5")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

