# Convolutional neural network for MNIST digit classification

import torch
import numpy as np
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from nn_model import ImageClassifier
from autoencoder_model import MNISTAutoencoder

# Load MNIST dataset
train = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

# Create data loaders with batching
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64, shuffle=False)

# Initialize model, optimizer, and loss function
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
if __name__ == "__main__":
    best_test_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(10):
        # Training phase
        clf.train()
        train_loss = 0.0
        
        for batch in train_loader:
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda')
            
            # Forward pass and loss calculation
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            
            # Backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()
            
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
                X, y = X.to('cuda'), y.to('cuda')
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
            with open('model_state.pth', 'wb') as f:
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
    
    clf.eval()
    num_classes = 10
    feature_dim = 64 * (28-6) * (28-6)
    
    # Collect features for each class
    class_features = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch in train_loader:
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda')
            features = clf.get_features(X)
            
            # Group features by class
            for i in range(num_classes):
                mask = (y == i)
                if mask.sum() > 0:
                    class_features[i].append(features[mask].cpu())
    
    # Compute class means (prototypes)
    class_means = {}
    for i in range(num_classes):
        if class_features[i]:
            all_features = torch.cat(class_features[i], dim=0)
            class_means[i] = all_features.mean(dim=0)
            print(f"Class {i}: {len(all_features)} samples")
    
    # Compute diagonal covariance (simpler, more robust for high dimensions)
    print("\nComputing diagonal covariance matrix...")
    all_features_centered = []
    for i in range(num_classes):
        if class_features[i]:
            features = torch.cat(class_features[i], dim=0)
            centered = features - class_means[i]
            all_features_centered.append(centered)
    
    all_features_centered = torch.cat(all_features_centered, dim=0)
    
    # Use diagonal covariance only (assume feature independence)
    # This is much more stable for high-dimensional spaces
    variance = torch.var(all_features_centered, dim=0)
    variance += 1e-4  # Small regularization
    
    # Precision is just 1/variance for diagonal covariance
    precision_diag = 1.0 / variance
    
    print(f"✓ Diagonal covariance computed: {variance.shape}")
    print(f"  Mean variance: {variance.mean().item():.4f}")
    print(f"  Min variance: {variance.min().item():.4f}")
    
    # Calibrate threshold on training data
    print("\nCalibrating threshold on training data...")
    all_distances = []
    with torch.no_grad():
        for i in range(num_classes):
            if class_features[i]:
                features = torch.cat(class_features[i], dim=0)
                # Sample only 50 random features per class for efficiency (500 total)
                num_samples = min(50, len(features))
                indices = torch.randperm(len(features))[:num_samples]
                
                print(f"  Class {i}: computing {num_samples} distances...", end='\r')
                for idx in indices:
                    feat = features[idx]
                    mean = class_means[i]
                    diff = feat - mean
                    # Diagonal Mahalanobis: sqrt(sum((x-μ)^2 / σ^2))
                    distance = torch.sqrt(torch.sum(diff**2 * precision_diag)).item()
                    all_distances.append(distance)
        
        print(f"  Computed {len(all_distances)} total distances" + " "*20)
    
    all_distances = np.array(all_distances)
    threshold_95 = np.percentile(all_distances, 95)
    threshold_99 = np.percentile(all_distances, 99)
    mean_dist = np.mean(all_distances)
    std_dist = np.std(all_distances)
    
    print(f"Distance statistics on training data:")
    print(f"  Mean: {mean_dist:.2f}")
    print(f"  Std: {std_dist:.2f}")
    print(f"  95th percentile: {threshold_95:.2f}")
    print(f"  99th percentile: {threshold_99:.2f}")
    print(f"\nRecommended threshold: {threshold_95:.2f} (captures 95% of training data)")
    
    # Save OOD detection parameters
    ood_params = {
        'class_means': class_means,
        'precision_diag': precision_diag,  # Diagonal precision instead of full matrix
        'feature_dim': feature_dim,
        'threshold_95': threshold_95,
        'threshold_99': threshold_99,
        'mean_distance': mean_dist,
        'std_distance': std_dist
    }
    
    with open('ood_params.pth', 'wb') as f:
        save(ood_params, f)
    print(f"\n✓ OOD detection parameters saved to ood_params.pth")
    print("  - Class prototypes (means) for all 10 digits")
    print("  - Precision matrix for Mahalanobis distance")
    print(f"  - Calibrated threshold: {threshold_95:.2f}")
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
    
    print("\nTraining autoencoder for 5 epochs...")
    for epoch in range(5):
        autoencoder.train()
        train_recon_loss = 0.0
        
        for batch in train_loader:
            X, y = batch  # Now we NEED labels for class-conditional training
            X = X.to('cuda')
            y = y.to('cuda')
            
            # Forward pass: reconstruct input conditioned on label
            reconstruction = autoencoder(X, y)
            loss = ae_loss_fn(reconstruction, X)
            
            # Backpropagation
            ae_opt.zero_grad()
            loss.backward()
            ae_opt.step()
            
            train_recon_loss += loss.item()
        
        train_recon_loss /= len(train_loader)
        
        # Evaluate on test set
        autoencoder.eval()
        test_recon_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                X, y = batch
                X = X.to('cuda')
                y = y.to('cuda')
                reconstruction = autoencoder(X, y)
                loss = ae_loss_fn(reconstruction, X)
                test_recon_loss += loss.item()
        
        test_recon_loss /= len(test_loader)
        
        print(f"Epoch {epoch}: Train Recon Loss = {train_recon_loss:.6f}, Test Recon Loss = {test_recon_loss:.6f}")
    
    # Calibrate reconstruction error threshold using predicted labels
    print("\nCalibrating reconstruction error threshold...")
    print("Using classifier predictions to determine which manifold to use...")
    autoencoder.eval()
    clf.eval()
    recon_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            X, y_true = batch
            X = X.to('cuda')
            
            # Get classifier predictions (simulating inference scenario)
            output = clf(X)
            y_pred = torch.argmax(output, dim=1)
            
            # Compute reconstruction error using PREDICTED class
            # This simulates real inference: "I think this is a 3, does it look like a 3?"
            errors = autoencoder.reconstruction_error(X, y_pred)
            recon_errors.extend(errors.cpu().tolist())
    
    recon_errors = np.array(recon_errors)
    recon_threshold_95 = np.percentile(recon_errors, 95)
    recon_threshold_99 = np.percentile(recon_errors, 99)
    recon_mean = np.mean(recon_errors)
    recon_std = np.std(recon_errors)
    
    print(f"\nReconstruction error statistics on test data:")
    print(f"  Mean: {recon_mean:.6f}")
    print(f"  Std: {recon_std:.6f}")
    print(f"  95th percentile: {recon_threshold_95:.6f}")
    print(f"  99th percentile: {recon_threshold_99:.6f}")
    print(f"\nRecommended threshold: {recon_threshold_95:.6f}")
    
    # Save autoencoder and threshold
    with open('autoencoder.pth', 'wb') as f:
        save({
            'model_state': autoencoder.state_dict(),
            'threshold_95': recon_threshold_95,
            'threshold_99': recon_threshold_99,
            'mean_error': recon_mean,
            'std_error': recon_std
        }, f)
    
    print(f"\n✓ Autoencoder saved to autoencoder.pth")
    print(f"  - Reconstruction threshold (95%): {recon_threshold_95:.6f}")
    print("  - Use as first gate before digit classifier")
    print("="*60)