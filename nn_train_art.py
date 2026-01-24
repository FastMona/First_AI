"""Training program for Fuzzy ART (Adaptive Resonance Theory) and autoencoder.

Trains ART classifier, class-conditional autoencoder, and computes OOD detection
parameters using template matching and vigilance-based learning.

Dashboard Menu: Called by Option 3 - "Train with ART"
"""

import torch
import numpy as np
import time
from torch import nn, save, load
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from nn_model_art import FuzzyARTClassifier
from autoencoder_model import MNISTAutoencoder
from config import Config

# Load MNIST dataset
full_train = datasets.MNIST(root='training_data', train=True, download=True, transform=ToTensor())
test = datasets.MNIST(root='training_data', train=False, download=True, transform=ToTensor())

# Split training data: 80% train, 20% validation
train_size = int(Config.TRAIN_RATIO * len(full_train))
val_size = len(full_train) - train_size

train, validation = random_split(full_train, [train_size, val_size], 
                                 generator=torch.Generator().manual_seed(Config.RANDOM_SEED))

print(f"Dataset split:")
print(f"  Training: {len(train)} samples (for model training)")
print(f"  Validation: {len(validation)} samples (for threshold calibration)")
print(f"  Test: {len(test)} samples (for final evaluation)")

# Create data loaders with GPU optimizations
# Note: ART trains sequentially, but we can optimize data loading and evaluation
TRAIN_BATCH_SIZE = 64   # Keep smaller for ART sequential training
EVAL_BATCH_SIZE = 256   # Larger for parallel evaluation/feature extraction
NUM_WORKERS = 4

train_loader = DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(validation, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

# Initialize Fuzzy ART classifier
art = FuzzyARTClassifier(
    input_dim=Config.INPUT_SIZE * Config.INPUT_SIZE,
    max_categories=Config.ART_MAX_CATEGORIES,
    vigilance=Config.ART_VIGILANCE,
    learning_rate=Config.ART_LEARNING_RATE,
    choice_alpha=Config.ART_CHOICE_ALPHA
).to(Config.DEVICE)

loss_fn = nn.CrossEntropyLoss()

def main():
    """Main training function for Fuzzy ART classifier and autoencoder"""
    
    print("\n" + "="*80)
    print("  Training Fuzzy Adaptive Resonance Theory (ART) Network".center(80))
    print("="*80)
    print(f"\nART Parameters:")
    print(f"  Max Categories: {Config.ART_MAX_CATEGORIES}")
    print(f"  Vigilance: {Config.ART_VIGILANCE}")
    print(f"  Learning Rate: {Config.ART_LEARNING_RATE}")
    print(f"  Device: {Config.get_device_info()}")
    print(f"\n🚀 GPU Optimizations:")
    print(f"  • Data workers: {NUM_WORKERS} (parallel loading)")
    print(f"  • Pinned memory: True (faster transfers)")
    print(f"  • Mixed precision: True (fp16 for evaluation)")
    print(f"  • Eval batch size: {EVAL_BATCH_SIZE} (parallelized)")
    print(f"  Note: ART trains sequentially (inherent to algorithm)")
    print("\n" + "="*80)
    
    # Training phase - ART learns incrementally
    print("\nPhase 1: ART Incremental Learning (Online Training)")
    print("-" * 80)
    
    art.train()
    
    # Train for multiple passes through the data
    num_passes = 3  # ART typically needs fewer epochs due to fast learning
    total_batches = len(train_loader)
    
    for pass_num in range(num_passes):
        print(f"\n{'='*80}")
        print(f"  PASS {pass_num + 1}/{num_passes} - Processing {len(train)} training samples")
        print(f"{'='*80}")
        
        total_samples = 0
        pass_start_time = time.time()
        batch_times = []
        
        for batch_idx, (X, y) in enumerate(train_loader):
            batch_start = time.time()
            X, y = X.to(Config.DEVICE, non_blocking=True), y.to(Config.DEVICE, non_blocking=True)
            
            # ART trains on individual patterns (sequential due to resonance search)
            for i in range(X.size(0)):
                art.train_pattern(X[i].view(-1), y[i])
                total_samples += 1
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Show progress every 50 batches (much more frequent)
            if (batch_idx + 1) % 50 == 0:
                avg_batch_time = np.mean(batch_times[-50:])
                samples_per_sec = (50 * TRAIN_BATCH_SIZE) / sum(batch_times[-50:])
                progress_pct = (batch_idx + 1) / total_batches * 100
                eta_batches = total_batches - (batch_idx + 1)
                eta_seconds = eta_batches * avg_batch_time
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                current_time = time.strftime("%H:%M")
                
                print(f"{progress_pct:.1f}% complete | Batch {batch_idx + 1}/{total_batches} | "
                      f"Samples Processed: {total_samples} | Speed: {samples_per_sec:.1f} samp/sec | "
                      f"Time remaining: {eta_min:02d}:{eta_sec:02d} | {current_time}")
        
        pass_time = time.time() - pass_start_time
        pass_min = int(pass_time // 60)
        pass_sec = int(pass_time % 60)
        
        print(f"\n✓ Pass {pass_num + 1} complete in {pass_min}m {pass_sec}s")
        print(f"  Total samples processed: {total_samples}")
        print(f"  Categories committed: {art.num_committed}/{Config.ART_MAX_CATEGORIES}")
        print(f"  Average speed: {total_samples/pass_time:.1f} samples/s")
        
        # Evaluate after each pass
        print(f"\n  Evaluating on test set...", end="", flush=True)
        eval_start = time.time()
        art.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(Config.DEVICE, non_blocking=True), y.to(Config.DEVICE, non_blocking=True)
                X = X.view(X.size(0), -1)
                
                # Get predictions with mixed precision
                with torch.amp.autocast('cuda'):
                    logits = art.predict(X)
                    _, predicted = torch.max(logits, 1)
                
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        eval_time = time.time() - eval_start
        accuracy = 100 * correct / total
        print(f" done in {eval_time:.1f}s")
        print(f"  Test Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
        
        art.train()
    
    # Save the trained ART model
    print(f"\n✓ Saving ART model to {Config.MODEL_PATH_ART}")
    with open(Config.MODEL_PATH_ART, 'wb') as f:
        save(art.state_dict(), f)
    
    # Compute OOD detection parameters
    print("\n" + "="*60)
    print("Computing OOD Detection Parameters for ART Model")
    print("="*60)
    print("Using ART category templates as feature representations")
    print("Calibrating on VALIDATION set (held-out data, not training)")
    
    art.eval()
    num_classes = Config.NUM_CLASSES
    feature_dim = art.coded_dim  # ART uses complement-coded dimension
    
    # Collect features for each class from TRAINING data
    print("\nStep 1: Computing class prototypes from TRAINING data...")
    print(f"  Processing {len(train_loader)} batches...", flush=True)
    class_features_train = {i: [] for i in range(num_classes)}
    
    step1_start = time.time()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(Config.DEVICE, non_blocking=True), y.to(Config.DEVICE, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                features = art.get_features(X)
            
            # Group features by class
            for i in range(num_classes):
                mask = (y == i)
                if mask.sum() > 0:
                    class_features_train[i].append(features[mask].cpu())
            
            # Progress indicator
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [{(batch_idx + 1) / len(train_loader) * 100:5.1f}%] Batch {batch_idx + 1}/{len(train_loader)}", end="\r", flush=True)
    
    print()  # New line after progress
    step1_time = time.time() - step1_start
    
    # Compute class means (prototypes)
    class_means = {}
    for i in range(num_classes):
        if class_features_train[i]:
            all_features = torch.cat(class_features_train[i], dim=0)
            class_means[i] = all_features.mean(dim=0)
            print(f"  Class {i}: {len(all_features)} training samples, mean computed")
    
    print(f"✓ Step 1 complete in {step1_time:.1f}s")
    
    # Compute covariance
    print("\nStep 2: Computing covariance matrix from TRAINING data...")
    step2_start = time.time()
    all_features_centered = []
    for i in range(num_classes):
        if class_features_train[i]:
            features = torch.cat(class_features_train[i], dim=0)
            centered = features - class_means[i]
            all_features_centered.append(centered)
            print(f"  Class {i}: {len(features)} samples centered", end="\r", flush=True)
    
    print()  # New line
    all_features_centered = torch.cat(all_features_centered, dim=0)
    variance = torch.var(all_features_centered, dim=0)
    variance += Config.COVARIANCE_REG
    precision_diag = 1.0 / variance
    
    step2_time = time.time() - step2_start
    print(f"✓ Step 2 complete in {step2_time:.1f}s")
    print(f"  Diagonal covariance computed: {variance.shape}")
    print(f"  Mean variance: {variance.mean().item():.4f}")
    
    # Calibrate thresholds on VALIDATION data
    print("\nStep 3: Calibrating class-conditional thresholds on VALIDATION data...")
    print(f"  Processing {len(val_loader)} batches...", flush=True)
    step3_start = time.time()
    class_distances = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X, y = X.to(Config.DEVICE, non_blocking=True), y.to(Config.DEVICE, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                features = art.get_features(X)
            
            for i in range(len(y)):
                feat = features[i].cpu()
                label = y[i].item()
                
                if label in class_means:
                    mean = class_means[label]
                    diff = feat - mean
                    distance = torch.sqrt(torch.sum(diff**2 * precision_diag)).item()
                    class_distances[label].append(distance)
            
            # Progress indicator
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(val_loader):
                print(f"  [{(batch_idx + 1) / len(val_loader) * 100:5.1f}%] Batch {batch_idx + 1}/{len(val_loader)}", end="\r", flush=True)
    
    print()  # New line
    step3_time = time.time() - step3_start
    print(f"✓ Step 3 complete in {step3_time:.1f}s")
    print(f"  Computed distances on {sum(len(v) for v in class_distances.values())} validation samples")
    
    # Compute per-class thresholds
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
    
    # Global statistics
    all_distances = [d for distances in class_distances.values() for d in distances]
    global_threshold_95 = np.percentile(all_distances, 95)
    global_mean = np.mean(all_distances)
    print(f"\nGlobal statistics (for reference):")
    print(f"  Mean: {global_mean:.2f}")
    print(f"  95th percentile: {global_threshold_95:.2f}")
    
    # Save OOD detection parameters
    ood_params = {
        'class_means': class_means,
        'precision_diag': precision_diag,
        'feature_dim': feature_dim,
        'model_type': 'art',  # Track which model type created these parameters
        'class_thresholds_90': class_thresholds_90,
        'class_thresholds_95': class_thresholds_95,
        'class_thresholds_99': class_thresholds_99,
        'class_mean_distances': class_mean_distances,
        'class_std_distances': class_std_distances,
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
    
    autoencoder = MNISTAutoencoder(
        latent_dim=Config.LATENT_DIM,
        embedding_dim=Config.EMBEDDING_DIM
    ).to(Config.DEVICE)
    
    from torch.optim import Adam
    ae_opt = Adam(autoencoder.parameters(), lr=Config.LEARNING_RATE)
    ae_loss_fn = nn.MSELoss()
    ae_scaler = torch.cuda.amp.GradScaler()
    
    print(f"\nTraining autoencoder for {Config.AE_EPOCHS} epochs...")
    for epoch in range(Config.AE_EPOCHS):
        epoch_start = time.time()
        autoencoder.train()
        train_recon_loss = 0.0
        
        print(f"  Epoch {epoch + 1}/{Config.AE_EPOCHS}: Training...", end="", flush=True)
        for X, y in train_loader:
            X = X.to(Config.DEVICE, non_blocking=True)
            y = y.to(Config.DEVICE, non_blocking=True)
            
            ae_opt.zero_grad()
            with torch.amp.autocast('cuda'):
                reconstruction = autoencoder(X, y)
                loss = ae_loss_fn(reconstruction, X)
            
            ae_scaler.scale(loss).backward()
            ae_scaler.step(ae_opt)
            ae_scaler.update()
            
            train_recon_loss += loss.item()
        
        train_recon_loss /= len(train_loader)
        
        # Evaluate on test set
        print(" Evaluating...", end="", flush=True)
        autoencoder.eval()
        test_recon_loss = 0.0
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(Config.DEVICE, non_blocking=True)
                y = y.to(Config.DEVICE, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    reconstruction = autoencoder(X, y)
                    loss = ae_loss_fn(reconstruction, X)
                test_recon_loss += loss.item()
        
        test_recon_loss /= len(test_loader)
        epoch_time = time.time() - epoch_start
        
        print(f" Done in {epoch_time:.1f}s")
        print(f"    Train Loss: {train_recon_loss:.6f} | Test Loss: {test_recon_loss:.6f}")
    
    # Calibrate reconstruction error threshold on VALIDATION data
    print("\nCalibrating reconstruction error threshold on VALIDATION data...")
    print(f"  Processing {len(val_loader)} batches...", end="", flush=True)
    calib_start = time.time()
    autoencoder.eval()
    art.eval()
    recon_errors = []
    
    with torch.no_grad():
        for X, y_true in val_loader:
            X = X.to(Config.DEVICE, non_blocking=True)
            
            # Get ART predictions with mixed precision
            with torch.amp.autocast('cuda'):
                X_flat = X.view(X.size(0), -1)
                output = art.predict(X_flat)
                y_pred = torch.argmax(output, dim=1)
                
                # Compute reconstruction error using PREDICTED class
                errors = autoencoder.reconstruction_error(X, y_pred)
            
            recon_errors.extend(errors.cpu().tolist())
    
    calib_time = time.time() - calib_start
    print(f" Done in {calib_time:.1f}s")
    
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
    
    print("\n" + "="*80)
    print("  ART Training Complete!".center(80))
    print("="*80)
    print(f"\nModel files saved:")
    print(f"  - {Config.MODEL_PATH_ART} (ART classifier)")
    print(f"  - {Config.AUTOENCODER_PATH} (Autoencoder)")
    print(f"  - {Config.OOD_PARAMS_PATH} (OOD detection parameters)")
    print("\nYou can now use detect.py or detect_batch.py for inference.")
    print("="*80)

if __name__ == "__main__":
    main()
