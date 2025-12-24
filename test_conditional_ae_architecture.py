"""
Quick test to verify the class-conditional autoencoder architecture works.
This tests the model can be instantiated and forward pass works correctly.
Run this BEFORE training to catch any issues.
"""

import torch
from autoencoder_model import MNISTAutoencoder

def test_architecture():
    print("Testing Class-Conditional Autoencoder Architecture")
    print("="*60)
    
    # Create model
    print("\n1. Creating model...")
    model = MNISTAutoencoder(latent_dim=64, num_classes=10, embedding_dim=16)
    print(f"   ✓ Model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    
    print(f"   Input shape: {images.shape}")
    print(f"   Labels: {labels.tolist()}")
    
    output = model(images, labels)
    print(f"   Output shape: {output.shape}")
    assert output.shape == images.shape, "Output shape mismatch!"
    print(f"   ✓ Forward pass successful")
    
    # Test reconstruction error
    print("\n3. Testing reconstruction_error()...")
    errors = model.reconstruction_error(images, labels)
    print(f"   Error shape: {errors.shape}")
    print(f"   Error values: {errors.tolist()}")
    assert errors.shape == (batch_size,), "Error shape mismatch!"
    print(f"   ✓ reconstruction_error() works")
    
    # Test class-specific error
    print("\n4. Testing get_class_specific_error()...")
    single_image = images[0:1]
    predicted_class = 3
    
    error = model.get_class_specific_error(single_image, predicted_class, return_all=False)
    print(f"   Error for class {predicted_class}: {error.item():.6f}")
    
    error, all_errors = model.get_class_specific_error(single_image, predicted_class, return_all=True)
    print(f"   Errors for all classes:")
    for class_idx, err in all_errors.items():
        print(f"     Class {class_idx}: {err:.6f}")
    print(f"   ✓ get_class_specific_error() works")
    
    # Test with different classes
    print("\n5. Testing reconstruction with different class labels...")
    single_image = torch.randn(1, 1, 28, 28)
    
    model.eval()
    with torch.no_grad():
        for class_idx in range(10):
            label = torch.tensor([class_idx])
            reconstruction = model(single_image, label)
            error = model.reconstruction_error(single_image, label)
            print(f"   Class {class_idx}: error = {error.item():.6f}")
    
    print(f"   ✓ All classes can be used for reconstruction")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print("\n6. Testing CUDA...")
        model_cuda = model.to('cuda')
        images_cuda = images.to('cuda')
        labels_cuda = labels.to('cuda')
        
        output_cuda = model_cuda(images_cuda, labels_cuda)
        errors_cuda = model_cuda.reconstruction_error(images_cuda, labels_cuda)
        
        print(f"   ✓ CUDA works")
    else:
        print("\n6. CUDA not available, skipping...")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("Architecture is ready for training.")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python nn_train.py")
    print("2. Run: python visualize_conditional_ae.py")
    print("3. Run: python test_accuracy.py")

if __name__ == "__main__":
    test_architecture()
