"""
Visualize Class-Conditional Autoencoder

This script demonstrates how the class-conditional autoencoder learns
separate manifolds for each digit class.

It shows:
1. Original image
2. Reconstruction using the CORRECT class manifold
3. Reconstruction using WRONG class manifolds
4. Reconstruction errors for all classes

This illustrates the biological perception model:
"I think this is a 3 — does it look like a 3?"
"""

import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from autoencoder_model import MNISTAutoencoder
from nn_model import ImageClassifier
import numpy as np

def visualize_conditional_reconstruction(autoencoder, classifier, test_dataset, num_samples=5):
    """
    Visualize how the autoencoder reconstructs images using different class manifolds.
    
    For each sample, show:
    - Original image
    - Classifier's prediction
    - Reconstruction using predicted class
    - Reconstruction errors for all 10 classes (as a bar chart)
    """
    autoencoder.eval()
    classifier.eval()
    
    fig = plt.figure(figsize=(18, 4 * num_samples))
    
    for sample_idx in range(num_samples):
        # Get a random sample
        idx = np.random.randint(0, len(test_dataset))
        image, true_label = test_dataset[idx]
        image_batch = image.unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            # Get classifier prediction
            output = classifier(image_batch)
            probs = torch.softmax(output, dim=1)[0]
            predicted_label = torch.argmax(probs).item()
            confidence = probs[predicted_label].item()
            
            # Get reconstruction errors for ALL classes
            all_errors = {}
            reconstructions = {}
            
            for class_idx in range(10):
                label_tensor = torch.tensor([class_idx], dtype=torch.long, device='cuda')
                reconstruction = autoencoder(image_batch, label_tensor)
                error = torch.mean((image_batch - reconstruction)**2).item()
                all_errors[class_idx] = error
                reconstructions[class_idx] = reconstruction.cpu().squeeze()
        
        # Plot original image
        ax = plt.subplot(num_samples, 13, sample_idx * 13 + 1)
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f'Original\nTrue: {true_label}', fontsize=10)
        ax.axis('off')
        
        # Plot classifier prediction
        ax = plt.subplot(num_samples, 13, sample_idx * 13 + 2)
        ax.text(0.5, 0.5, f'Pred: {predicted_label}\nConf: {confidence:.2f}', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Plot reconstructions for each class
        for class_idx in range(10):
            ax = plt.subplot(num_samples, 13, sample_idx * 13 + 3 + class_idx)
            ax.imshow(reconstructions[class_idx].squeeze(), cmap='gray')
            
            # Highlight the predicted class in green, true class in blue
            border_color = 'green' if class_idx == predicted_label else ('blue' if class_idx == true_label else 'none')
            linewidth = 3 if class_idx in [predicted_label, true_label] else 0
            
            title = f'Class {class_idx}\nErr: {all_errors[class_idx]:.4f}'
            ax.set_title(title, fontsize=8, color=border_color if border_color != 'none' else 'black')
            ax.axis('off')
            
            if linewidth > 0:
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(linewidth)
                    spine.set_visible(True)
        
        # Plot error bar chart
        ax = plt.subplot(num_samples, 13, sample_idx * 13 + 13)
        classes = list(range(10))
        errors = [all_errors[c] for c in classes]
        colors = ['green' if c == predicted_label else ('blue' if c == true_label else 'gray') for c in classes]
        ax.bar(classes, errors, color=colors, alpha=0.7)
        ax.set_xlabel('Class', fontsize=8)
        ax.set_ylabel('Error', fontsize=8)
        ax.set_title('Recon Errors', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(classes)
    
    plt.tight_layout()
    plt.savefig('conditional_ae_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to conditional_ae_visualization.png")
    plt.show()


def analyze_manifold_separation(autoencoder, test_dataset, num_samples=1000):
    """
    Analyze how well the autoencoder separates different digit manifolds.
    
    For each digit class, compute:
    - Mean reconstruction error using correct manifold
    - Mean reconstruction error using wrong manifolds
    """
    autoencoder.eval()
    
    # Sample images from each class
    class_samples = {i: [] for i in range(10)}
    
    for image, label in test_dataset:
        if len(class_samples[label]) < num_samples // 10:
            class_samples[label].append(image)
        
        if all(len(samples) >= num_samples // 10 for samples in class_samples.values()):
            break
    
    print("\nManifold Separation Analysis")
    print("="*70)
    print("For each true digit, comparing reconstruction error using:")
    print("  - Correct manifold (same as true label)")
    print("  - Wrong manifolds (all other labels)")
    print("="*70)
    
    results = []
    
    with torch.no_grad():
        for true_class in range(10):
            samples = class_samples[true_class]
            correct_errors = []
            wrong_errors = []
            
            for image in samples:
                image_batch = image.unsqueeze(0).to('cuda')
                
                for manifold_class in range(10):
                    label_tensor = torch.tensor([manifold_class], dtype=torch.long, device='cuda')
                    error = autoencoder.reconstruction_error(image_batch, label_tensor).item()
                    
                    if manifold_class == true_class:
                        correct_errors.append(error)
                    else:
                        wrong_errors.append(error)
            
            correct_mean = np.mean(correct_errors)
            wrong_mean = np.mean(wrong_errors)
            separation = wrong_mean - correct_mean
            ratio = wrong_mean / correct_mean if correct_mean > 0 else float('inf')
            
            results.append({
                'class': true_class,
                'correct': correct_mean,
                'wrong': wrong_mean,
                'separation': separation,
                'ratio': ratio
            })
            
            print(f"Digit {true_class}: Correct={correct_mean:.6f}, Wrong={wrong_mean:.6f}, "
                  f"Separation={separation:.6f}, Ratio={ratio:.2f}x")
    
    print("="*70)
    avg_separation = np.mean([r['separation'] for r in results])
    avg_ratio = np.mean([r['ratio'] for r in results])
    print(f"Average separation: {avg_separation:.6f}")
    print(f"Average ratio (wrong/correct): {avg_ratio:.2f}x")
    print("\nHigher values indicate better manifold separation!")


if __name__ == "__main__":
    # Load models
    print("Loading models...")
    
    # Load classifier
    classifier = ImageClassifier().to('cuda')
    with open('model_state.pth', 'rb') as f:
        classifier.load_state_dict(torch.load(f))
    classifier.eval()
    
    # Load autoencoder
    with open('autoencoder.pth', 'rb') as f:
        ae_data = torch.load(f)
    autoencoder = MNISTAutoencoder(latent_dim=64).to('cuda')
    autoencoder.load_state_dict(ae_data['model_state'])
    autoencoder.eval()
    
    print("Models loaded successfully!")
    
    # Load test dataset
    test_dataset = MNIST(root='./data', train=False, download=True, transform=ToTensor())
    
    print("\n" + "="*70)
    print("Class-Conditional Autoencoder Visualization")
    print("="*70)
    print("\nThis demonstrates biological perception:")
    print('"I think this is a 3 — does it look like a 3?"')
    print("\nGreen border = Predicted class manifold")
    print("Blue border = True class manifold")
    print("\nThe autoencoder should have LOW error for the correct class")
    print("and HIGH error for incorrect classes.")
    print("="*70)
    
    # Visualize reconstructions
    visualize_conditional_reconstruction(autoencoder, classifier, test_dataset, num_samples=5)
    
    # Analyze manifold separation
    analyze_manifold_separation(autoencoder, test_dataset, num_samples=1000)
