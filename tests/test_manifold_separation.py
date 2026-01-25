"""Unit tests for autoencoder manifold separation (class-conditional AE).

Tests that the autoencoder properly learns separate class-conditional manifolds,
which is critical for Stage 1 OOD detection (reconstruction error gate).
"""

import pytest
import numpy as np

# Skip entire module if torch not available
torch = pytest.importorskip("torch")

from pathlib import Path

from autoencoder_model import MNISTAutoencoder
from nn_model_cnn import ImageClassifier
from config import Config


class TestManifoldSeparation:
    """Test class-conditional autoencoder manifold separation."""
    
    @pytest.fixture(scope="class")
    def models(self):
        """Load pretrained models for testing."""
        # Check if models exist
        if not Config.MODEL_PATH.exists():
            pytest.skip("CNN model not trained - skipping manifold tests")
        if not Config.AUTOENCODER_PATH.exists():
            pytest.skip("Autoencoder not trained - skipping manifold tests")
        
        # Determine device (CPU if CUDA unavailable)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load classifier
        classifier = ImageClassifier().to(device)
        with open(Config.MODEL_PATH, 'rb') as f:
            classifier.load_state_dict(torch.load(f, map_location=device, weights_only=False))
        classifier.eval()
        
        # Load autoencoder
        with open(Config.AUTOENCODER_PATH, 'rb') as f:
            ae_data = torch.load(f, map_location=device, weights_only=False)
        autoencoder = MNISTAutoencoder(latent_dim=Config.LATENT_DIM).to(device)
        autoencoder.load_state_dict(ae_data['model_state'])
        autoencoder.eval()
        
        return {
            'classifier': classifier,
            'autoencoder': autoencoder,
            'device': device
        }
    
    @pytest.fixture
    def test_dataset(self):
        """Load MNIST test dataset."""
        pytest.importorskip("torchvision")
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        
        return MNIST(root='./training_data', train=False, download=True, transform=ToTensor())
    
    def test_autoencoder_loaded(self, models):
        """Test that autoencoder is properly loaded."""
        ae = models['autoencoder']
        assert ae is not None
        assert hasattr(ae, 'forward')
        assert hasattr(ae, 'encoder')
        assert hasattr(ae, 'decoder')
    
    def test_reconstruction_error_computation(self, models, test_dataset):
        """Test that reconstruction errors are computed correctly."""
        ae = models['autoencoder']
        device = models['device']
        
        # Get a sample image
        image, true_label = test_dataset[0]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Test reconstruction with different class manifolds
            errors = []
            for class_idx in range(10):
                label_tensor = torch.tensor([class_idx], dtype=torch.long, device=device)
                reconstruction = ae(image_batch, label_tensor)
                error = torch.mean((image_batch - reconstruction)**2).item()
                errors.append(error)
            
            # All errors should be positive
            assert all(e > 0 for e in errors), "Reconstruction errors should be positive"
            # Errors should be finite
            assert all(np.isfinite(e) for e in errors), "Reconstruction errors should be finite"
    
    def test_manifold_separation_ratio(self, models, test_dataset):
        """Test that manifolds show separation (wrong manifolds have higher error).
        
        For digits of class X, reconstruction error should be:
        - LOW when using class X manifold
        - HIGH when using wrong class manifolds
        
        Threshold: ratio of wrong/correct should be at least 1.2x
        """
        ae = models['autoencoder']
        device = models['device']
        
        # Sample ~100 test images from each class
        samples_per_class = 100
        class_samples = {i: [] for i in range(10)}
        
        for image, label in test_dataset:
            if len(class_samples[label]) < samples_per_class:
                class_samples[label].append(image.to(device))
            
            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break
        
        separation_ratios = []
        
        with torch.no_grad():
            for true_class in range(10):
                if not class_samples[true_class]:
                    continue
                
                correct_errors = []
                wrong_errors = []
                
                for image in class_samples[true_class]:
                    image_batch = image.unsqueeze(0)
                    
                    # Error with correct manifold
                    label_tensor = torch.tensor([true_class], dtype=torch.long, device=device)
                    reconstruction = ae(image_batch, label_tensor)
                    error = torch.mean((image_batch - reconstruction)**2).item()
                    correct_errors.append(error)
                    
                    # Errors with wrong manifolds
                    for wrong_class in range(10):
                        if wrong_class != true_class:
                            label_tensor = torch.tensor([wrong_class], dtype=torch.long, device=device)
                            reconstruction = ae(image_batch, label_tensor)
                            error = torch.mean((image_batch - reconstruction)**2).item()
                            wrong_errors.append(error)
                
                correct_mean = np.mean(correct_errors)
                wrong_mean = np.mean(wrong_errors)
                
                if correct_mean > 0:
                    ratio = wrong_mean / correct_mean
                    separation_ratios.append(ratio)
        
        # Test: average separation ratio should be at least 1.2x
        avg_ratio = np.mean(separation_ratios)
        assert avg_ratio >= 1.2, (
            f"Manifold separation ratio {avg_ratio:.2f}x is too low (threshold: 1.2x). "
            "Autoencoder may not be learning proper class-conditional manifolds."
        )
    
    def test_class_manifold_distinctiveness(self, models, test_dataset):
        """Test that each class has a distinctive manifold.
        
        For a single image, reconstruction errors should vary significantly
        across different class manifolds.
        """
        ae = models['autoencoder']
        device = models['device']
        
        # Get multiple samples
        test_indices = [0, 10, 20, 30, 40]
        
        with torch.no_grad():
            for idx in test_indices:
                image, true_label = test_dataset[idx]
                image_batch = image.unsqueeze(0).to(device)
                
                errors = []
                for class_idx in range(10):
                    label_tensor = torch.tensor([class_idx], dtype=torch.long, device=device)
                    reconstruction = ae(image_batch, label_tensor)
                    error = torch.mean((image_batch - reconstruction)**2).item()
                    errors.append(error)
                
                # Standard deviation of errors across manifolds should be significant
                error_std = np.std(errors)
                error_mean = np.mean(errors)
                
                # Coefficient of variation should be at least 0.10 (10%)
                cv = error_std / error_mean if error_mean > 0 else 0
                assert cv >= 0.10, (
                    f"Manifold distinctiveness CV {cv:.2f} is too low (threshold: 0.10). "
                    "Class manifolds may not be sufficiently distinct."
                )
    
    def test_correct_manifold_lowest_error(self, models, test_dataset):
        """Test that correct class manifold typically has lowest error.
        
        For ~80% of test samples, the correct class should have the lowest
        reconstruction error compared to all other classes.
        """
        ae = models['autoencoder']
        device = models['device']
        
        # Test first 100 samples
        num_samples = 100
        correct_predictions = 0
        
        with torch.no_grad():
            for idx in range(min(num_samples, len(test_dataset))):
                image, true_label = test_dataset[idx]
                image_batch = image.unsqueeze(0).to(device)
                
                errors = []
                for class_idx in range(10):
                    label_tensor = torch.tensor([class_idx], dtype=torch.long, device=device)
                    reconstruction = ae(image_batch, label_tensor)
                    error = torch.mean((image_batch - reconstruction)**2).item()
                    errors.append(error)
                
                # Check if correct class has minimum error
                predicted_class = np.argmin(errors)
                if predicted_class == true_label:
                    correct_predictions += 1
        
        accuracy = correct_predictions / num_samples
        assert accuracy >= 0.7, (
            f"Manifold accuracy {accuracy:.1%} is too low (threshold: 70%). "
            "Autoencoder may not be learning meaningful class-conditional structure."
        )


def test_manifold_separation_integration(test_dataset=None):
    """Integration test: full manifold separation analysis.
    
    This test can be run standalone without pytest fixtures.
    """
    pytest.importorskip("torch")
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    
    # Skip if models not trained
    if not Config.MODEL_PATH.exists() or not Config.AUTOENCODER_PATH.exists():
        pytest.skip("Models not trained - skipping integration test")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    classifier = ImageClassifier().to(device)
    with open(Config.MODEL_PATH, 'rb') as f:
        classifier.load_state_dict(torch.load(f, map_location=device, weights_only=False))
    classifier.eval()
    
    with open(Config.AUTOENCODER_PATH, 'rb') as f:
        ae_data = torch.load(f, map_location=device, weights_only=False)
    autoencoder = MNISTAutoencoder(latent_dim=Config.LATENT_DIM).to(device)
    autoencoder.load_state_dict(ae_data['model_state'])
    autoencoder.eval()
    
    # Load test dataset if not provided
    if test_dataset is None:
        test_dataset = MNIST(root='./training_data', train=False, download=True, transform=ToTensor())
    
    # Quick validation: check that separation exists
    samples_per_class = 50
    class_samples = {i: [] for i in range(10)}
    
    for image, label in test_dataset:
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(image.to(device))
        
        if all(len(samples) >= samples_per_class for samples in class_samples.values()):
            break
    
    separation_ratios = []
    
    with torch.no_grad():
        for true_class in range(10):
            if not class_samples[true_class]:
                continue
            
            correct_errors = []
            wrong_errors = []
            
            for image in class_samples[true_class][:10]:  # Sample subset for speed
                image_batch = image.unsqueeze(0)
                
                # Correct manifold
                label_tensor = torch.tensor([true_class], dtype=torch.long, device=device)
                reconstruction = autoencoder(image_batch, label_tensor)
                error = torch.mean((image_batch - reconstruction)**2).item()
                correct_errors.append(error)
                
                # Wrong manifolds (sample 3 random wrong classes)
                for _ in range(3):
                    wrong_class = np.random.choice([c for c in range(10) if c != true_class])
                    label_tensor = torch.tensor([wrong_class], dtype=torch.long, device=device)
                    reconstruction = autoencoder(image_batch, label_tensor)
                    error = torch.mean((image_batch - reconstruction)**2).item()
                    wrong_errors.append(error)
            
            correct_mean = np.mean(correct_errors)
            wrong_mean = np.mean(wrong_errors)
            
            if correct_mean > 0:
                ratio = wrong_mean / correct_mean
                separation_ratios.append(ratio)
    
    avg_ratio = np.mean(separation_ratios)
    assert avg_ratio >= 1.2, f"Manifold separation too weak: {avg_ratio:.2f}x (need 1.2x+)"
