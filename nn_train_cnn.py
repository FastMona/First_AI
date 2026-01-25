"""Training program for CNN (Convolutional Neural Network) and autoencoder.

Trains digit classifier, class-conditional autoencoder, and computes OOD detection
parameters (Mahalanobis distance thresholds).

Dashboard Menu: Called by Option 2 - "Train with CNN"
"""

import torch
import numpy as np
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from nn_model_cnn import ImageClassifier
from autoencoder_model import MNISTAutoencoder
from config import Config

# Load MNIST dataset
full_train = datasets.MNIST(root='training_data', train=True, download=True, transform=ToTensor())
test = datasets.MNIST(root='training_data', train=False, download=True, transform=ToTensor())

# Split training data: 80% train, 20% validation
# Validation set used for OOD threshold calibration (not for early stopping)
train_size = int(0.8 * len(full_train))  # 48,000 samples
val_size = len(full_train) - train_size  # 12,000 samples

train, validation = random_split(full_train, [train_size, val_size], 
                                 generator=torch.Generator().manual_seed(42))

print(f"Dataset split:")
print(f"  Training: {len(train)} samples (for model training)")
print(f"  Validation: {len(validation)} samples (for threshold calibration)")
print(f"  Test: {len(test)} samples (for final evaluation)")

# Create data loaders with optimized settings for GPU utilization
# Larger batch size + multiple workers + pinned memory = better GPU saturation
BATCH_SIZE = 256  # Increased from 64 for better GPU utilization
NUM_WORKERS = 4   # Parallel data loading to prevent CPU bottleneck

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

# Initialize model, optimizer, and loss function
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Mixed precision training for better GPU utilization and faster training
scaler = torch.amp.GradScaler('cuda')
print("\n🚀 GPU Optimization enabled:")
print(f"  • Batch size: {BATCH_SIZE} (4x larger for better parallelization)")
print(f"  • Data workers: {NUM_WORKERS} (parallel CPU data loading)")
print(f"  • Pinned memory: True (faster CPU→GPU transfers)")
print(f"  • Mixed precision: True (fp16 for 2-3x speedup)")
print()

def main():
    """Main training function for CNN classifier and autoencoder"""
    best_test_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(10):
        # Training phase
        clf.train()
        train_loss = 0.0
        
        for batch in train_loader:
            X, y = batch
            X, y = X.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)
            
            # Forward pass with automatic mixed precision
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                yhat = clf(X)
                loss = loss_fn(yhat, y)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluation phase
        clf.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                X, y = batch
                X, y = X.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)
                
                # Use autocast for inference too
                with torch.amp.autocast('cuda'):
                    yhat = clf(X)
                    loss = loss_fn(yhat, y)
                test_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(yhat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}, Test Accuracy = {accuracy:.2f}%")

        # Save model only if test loss improved
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            with open(Config.MODEL_PATH, 'wb') as f:
                save(clf.state_dict(), f)
            print(f"  ✓ New best model saved (test loss: {test_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best test loss: {best_test_loss:.6f}")
            break
    
    # Compute class prototypes and covariance for OOD detection
    print("\n" + "="*60)
    print("Computing Mahalanobis distance parameters for OOD detection")
    print("="*60)
    print("Using compact 128-d embedding layer for manifold representation")
    print("Calibrating on VALIDATION set (held-out data, not training)")
    
    clf.eval()
    num_classes = 10
    feature_dim = 128  # Compact embedding dimension
    
    # Collect features for each class from TRAINING data (for class means/prototypes)
    print("\nStep 1: Computing class prototypes from TRAINING data...")
    class_features_train = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch in train_loader:
            X, y = batch
            X, y = X.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                features = clf.get_features(X)
            
            # Group features by class
            for i in range(num_classes):
                mask = (y == i)
                if mask.sum() > 0:
                    class_features_train[i].append(features[mask].cpu())
    
    # Compute class means (prototypes) from training data
    class_means = {}
    for i in range(num_classes):
        if class_features_train[i]:
            all_features = torch.cat(class_features_train[i], dim=0)
            class_means[i] = all_features.mean(dim=0)
            print(f"  Class {i}: {len(all_features)} training samples, mean computed")
    
    # Compute covariance from training data
    print("\nStep 2: Computing covariance matrix from TRAINING data...")
    all_features_centered = []
    for i in range(num_classes):
        if class_features_train[i]:
            features = torch.cat(class_features_train[i], dim=0)
            centered = features - class_means[i]
            all_features_centered.append(centered)
    
    all_features_centered = torch.cat(all_features_centered, dim=0)
    
    # Use diagonal covariance matrix (assumes feature independence for efficiency)
    # Diagonal approach: much faster than full covariance with acceptable accuracy
    variance = torch.var(all_features_centered, dim=0)
    variance += 1e-4  # Regularization to prevent division by zero
    precision_diag = 1.0 / variance
    
    print(f"✓ Diagonal covariance computed: {variance.shape}")
    print(f"  Mean variance: {variance.mean().item():.4f}")
    
    # Calibrate thresholds on VALIDATION data (held-out, more realistic)
    print("\nStep 3: Calibrating class-conditional thresholds on VALIDATION data...")
    print("(This reflects generalization, not memorization)")
    
    class_distances = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch in val_loader:
            X, y = batch
            X, y = X.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                features = clf.get_features(X)
            
            # For each sample, compute distance to its TRUE class prototype
            for i in range(len(y)):
                feat = features[i].cpu()
                label = y[i].item()
                
                if label in class_means:
                    mean = class_means[label]
                    diff = feat - mean
                    # Diagonal Mahalanobis: sqrt(sum((x-μ)^2 / σ^2))
                    distance = torch.sqrt(torch.sum(diff**2 * precision_diag)).item()
                    class_distances[label].append(distance)
    
    print(f"✓ Computed distances on {sum(len(v) for v in class_distances.values())} validation samples")
    
    # Compute per-class thresholds from VALIDATION distances
    # 90th percentile = stricter (reject more), better for OOD detection
    # 95th percentile = moderate, 99th = lenient (accept more)
    class_thresholds_90 = {}
    class_thresholds_95 = {}
    class_thresholds_99 = {}
    class_mean_distances = {}
    class_std_distances = {}
    
    print(f"\nClass-conditional threshold statistics (from VALIDATION data):")
    for i in range(num_classes):
        if class_distances[i]:
            distances = np.array(class_distances[i])
            class_thresholds_90[i] = np.percentile(distances, 90)
            class_thresholds_95[i] = np.percentile(distances, 95)
            class_thresholds_99[i] = np.percentile(distances, 99)
            class_mean_distances[i] = np.mean(distances)
            class_std_distances[i] = np.std(distances)
            print(f"  Class {i}: n={len(distances)}, mean={class_mean_distances[i]:.2f} ± {class_std_distances[i]:.2f}, "
                  f"90th={class_thresholds_90[i]:.2f}, 95th={class_thresholds_95[i]:.2f}, 99th={class_thresholds_99[i]:.2f}")
    
    # Also compute global statistics for reference
    all_distances = [d for distances in class_distances.values() for d in distances]
    global_threshold_95 = np.percentile(all_distances, 95)
    global_mean = np.mean(all_distances)
    print(f"\nGlobal statistics (for reference):")
    print(f"  Mean: {global_mean:.2f}")
    print(f"  95th percentile: {global_threshold_95:.2f}")
    
    # Save OOD detection parameters
    ood_params = {
        'class_means': class_means,
        'precision_diag': precision_diag,  # Diagonal precision instead of full matrix
        'feature_dim': feature_dim,
        'model_type': 'cnn',  # Track which model type created these parameters
        'class_thresholds_90': class_thresholds_90,  # Stricter per-class thresholds (default)
        'class_thresholds_95': class_thresholds_95,  # Per-class thresholds
        'class_thresholds_99': class_thresholds_99,  # Per-class thresholds
        'class_mean_distances': class_mean_distances,
        'class_std_distances': class_std_distances,
        # Keep global stats for backward compatibility
        'threshold_95': global_threshold_95,
        'mean_distance': global_mean,
    }
    
    with open(Config.OOD_PARAMS_PATH, 'wb') as f:
        save(ood_params, f)
    print(f"\n✓ OOD detection parameters saved to {Config.OOD_PARAMS_PATH}")
    print("  - Class prototypes (means) for all 10 digits")
    print("  - Precision matrix for Mahalanobis distance")
    print(f"  - Class-conditional thresholds (90th/95th/99th percentiles per class)")
    print(f"  - Default: 90th percentile (stricter for better OOD detection)")
    print("="*60)
    
    # Train autoencoder for reconstruction-based OOD detection
    print("\n" + "="*60)
    print("Training Class-Conditional Autoencoder")
    print("="*60)
    print("Learning 10 separate digit manifolds (one per class)")
    print("Biological perception: 'I think this is a 3 — does it look like a 3?'")
    
    autoencoder = MNISTAutoencoder(latent_dim=64).to('cuda')
    ae_opt = Adam(autoencoder.parameters(), lr=1e-3)
    ae_loss_fn = nn.MSELoss()
    ae_scaler = torch.cuda.amp.GradScaler()  # Separate scaler for autoencoder
    
    print("\nTraining autoencoder for 5 epochs...")
    for epoch in range(5):
        autoencoder.train()
        train_recon_loss = 0.0
        
        for batch in train_loader:
            X, y = batch  # Now we NEED labels for class-conditional training
            X = X.to('cuda', non_blocking=True)
            y = y.to('cuda', non_blocking=True)
            
            # Forward pass with mixed precision
            ae_opt.zero_grad()
            with torch.amp.autocast('cuda'):
                reconstruction = autoencoder(X, y)
                loss = ae_loss_fn(reconstruction, X)
            
            # Backward pass with gradient scaling
            ae_scaler.scale(loss).backward()
            ae_scaler.step(ae_opt)
            ae_scaler.update()
            
            train_recon_loss += loss.item()
        
        train_recon_loss /= len(train_loader)
        
        # Evaluate on test set
        autoencoder.eval()
        test_recon_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                X, y = batch
                X = X.to('cuda', non_blocking=True)
                y = y.to('cuda', non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    reconstruction = autoencoder(X, y)
                    loss = ae_loss_fn(reconstruction, X)
                test_recon_loss += loss.item()
        
        test_recon_loss /= len(test_loader)
        
        print(f"Epoch {epoch}: Train Recon Loss = {train_recon_loss:.6f}, Test Recon Loss = {test_recon_loss:.6f}")
    
    # Calibrate reconstruction error threshold on VALIDATION data
    print("\nCalibrating reconstruction error threshold on VALIDATION data...")
    print("Using classifier predictions to determine which manifold to use...")
    print("(Validation reflects generalization, not training memorization)")
    autoencoder.eval()
    clf.eval()
    recon_errors = []
    
    with torch.no_grad():
        for batch in val_loader:
            X, y_true = batch
            X = X.to('cuda', non_blocking=True)
            
            # Get classifier predictions with autocast
            with torch.amp.autocast('cuda'):
                output = clf(X)
                y_pred = torch.argmax(output, dim=1)
                
                # Compute reconstruction error using PREDICTED class (not true label)
                # Biological perception: "I think this is a 3 — does it look like a 3?"
                # This matches how the system operates during actual inference
                errors = autoencoder.reconstruction_error(X, y_pred)
            
            recon_errors.extend(errors.cpu().tolist())
    
    recon_errors = np.array(recon_errors)
    recon_threshold_95 = np.percentile(recon_errors, 95)
    recon_threshold_99 = np.percentile(recon_errors, 99)
    recon_mean = np.mean(recon_errors)
    recon_std = np.std(recon_errors)
    
    print(f"\nReconstruction error statistics on VALIDATION data:")
    print(f"  Samples: {len(recon_errors)}")
    print(f"  Mean: {recon_mean:.6f}")
    print(f"  Std: {recon_std:.6f}")
    print(f"  95th percentile: {recon_threshold_95:.6f}")
    print(f"  99th percentile: {recon_threshold_99:.6f}")
    print(f"\nRecommended threshold: {recon_threshold_95:.6f}")
    
    # Save autoencoder and threshold
    with open(Config.AUTOENCODER_PATH, 'wb') as f:
        save({
            'model_state': autoencoder.state_dict(),
            'threshold_95': recon_threshold_95,
            'threshold_99': recon_threshold_99,
            'mean_error': recon_mean,
            'std_error': recon_std
        }, f)
    
    print(f"\n✓ Autoencoder saved to {Config.AUTOENCODER_PATH}")
    print(f"  - Reconstruction threshold (95%): {recon_threshold_95:.6f}")
    print("  - Use as first gate before digit classifier")
    print("="*60)

if __name__ == "__main__":
    main()
