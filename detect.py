# Digit detection program for MNIST model
# Loads trained model and predicts digits from image files

import torch
import numpy as np
from PIL import Image
from torch import nn, load
from torchvision.transforms import ToTensor
from nn_model import ImageClassifier
from autoencoder_model import MNISTAutoencoder
from ood_detector import MahalanobisOODDetector

def predict_image(image_path, model, autoencoder, ood_detector, ae_threshold, confidence_threshold=0.80):
    """
    Predict the digit in an image with two-stage OOD detection
    
    Stage 1: Autoencoder reconstruction error (rejects non-digits)
    Stage 2: Mahalanobis distance to class prototypes (refines detection)
    
    Args:
        image_path: Path to the image file
        model: Trained ImageClassifier model
        autoencoder: Trained autoencoder
        ood_detector: MahalanobisOODDetector instance
        ae_threshold: Reconstruction error threshold
        confidence_threshold: Minimum classifier confidence
    
    Returns:
        prediction, confidence, belongs, recon_error, mahal_distance, threshold
    """
    img = Image.open(image_path)
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        # Stage 1: Check reconstruction error
        recon_error = autoencoder.reconstruction_error(img_tensor).item()
        
        # If reconstruction fails, reject immediately
        if recon_error > ae_threshold:
            # Still get prediction for display purposes
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            prediction = torch.argmax(probs).item()
            confidence = probs[prediction].item()
            return prediction, confidence, False, recon_error, None, ae_threshold, "reconstruction"
        
        # Stage 2: Passed reconstruction, now check Mahalanobis distance
        features = model.get_features(img_tensor)
        output = model(img_tensor)
        
        probs = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probs).item()
        confidence = probs[prediction].item()
        
        belongs, mahal_distance, min_distance, nearest_class, all_distances = ood_detector.detect(
            features[0], prediction
        )
        
        mahal_threshold = ood_detector.threshold_95
    
    return prediction, confidence, belongs, recon_error, min_distance, mahal_threshold, "mahalanobis"

def main():
    print("="*60)
    print("MNIST Digit Detector with 2-Stage OOD Detection")
    print("="*60)
    
    # Load the trained model
    print("\nLoading trained model from model_state.pth...")
    clf = ImageClassifier().to('cuda')
    
    try:
        with open('model_state.pth', 'rb') as f:
            clf.load_state_dict(load(f))
        clf.eval()
        print("✓ Model loaded successfully!")
    except FileNotFoundError:
        print("Error: model_state.pth not found!")
        print("Please train the model first using nn_train.py")
        return
    
    # Load autoencoder
    try:
        with open('autoencoder.pth', 'rb') as f:
            ae_data = load(f)
        autoencoder = MNISTAutoencoder(latent_dim=64).to('cuda')
        autoencoder.load_state_dict(ae_data['model_state'])
        autoencoder.eval()
        ae_threshold = ae_data['threshold_95']
        print(f"✓ Autoencoder loaded (threshold: {ae_threshold:.6f})")
    except FileNotFoundError:
        print("Error: autoencoder.pth not found!")
        print("Please train the model first to generate autoencoder")
        return
    
    # Load OOD detector
    try:
        ood_detector = MahalanobisOODDetector('ood_params.pth')
    except FileNotFoundError:
        print("Error: ood_params.pth not found!")
        print("Please train the model first to generate OOD parameters")
        return
    
    # Get image filename from user
    print("\n" + "-"*60)
    image_path = input("Enter image filename (e.g., test_images/img_1.jpg): ")
    
    try:
        # Make prediction with two-stage OOD detection
        prediction, confidence, belongs, recon_error, distance, threshold, rejection_stage = predict_image(
            image_path, clf, autoencoder, ood_detector, ae_threshold
        )
        
        # Display results
        print("\n" + "="*60)
        
        if not belongs:
            if rejection_stage == "reconstruction":
                print("❌ REJECTED AT STAGE 1: RECONSTRUCTION ERROR TOO HIGH")
                print(f"\nThis image cannot be reconstructed as a digit.")
                print(f"Reconstruction error: {recon_error:.6f} (threshold: {threshold:.6f})")
                print(f"\nClassifier's guess: {prediction} ({confidence*100:.1f}%)")
                print("\n💡 Stage 1 Gate: Autoencoder REJECTED this as NOT a digit")
                print("   The autoencoder learned only digits, so it can't recreate this.")
            else:
                print("❌ REJECTED AT STAGE 2: MAHALANOBIS DISTANCE TOO HIGH")
                print(f"\nReconstruction error: {recon_error:.6f} ✓ (passed stage 1)")
                print(f"Mahalanobis distance: {distance:.2f} ✗ (threshold: {threshold:.2f})")
                print(f"\nClassifier's guess: {prediction} ({confidence*100:.1f}%)")
                print("\n💡 Stage 1 passed, but Stage 2 Mahalanobis distance REJECTED")
                print("   Image reconstructs OK but doesn't match digit prototypes.")
        else:
            print(f"✓ PASSED BOTH STAGES - VALID DIGIT")
            print(f"\n🔢 Predicted Digit: {prediction}")
            print(f"   Confidence: {confidence*100:.1f}%")
            print(f"\nStage 1 - Reconstruction error: {recon_error:.6f} ✓ (threshold: {ae_threshold:.6f})")
            print(f"Stage 2 - Mahalanobis distance: {distance:.2f} ✓ (threshold: {threshold:.2f})")
            
            relative_recon = recon_error / ae_threshold * 100
            relative_mahal = distance / threshold * 100
            
            if relative_recon < 50 and relative_mahal < 50:
                print(f"\n💪 Excellent digit - very typical example!")
            elif relative_recon < 75 and relative_mahal < 75:
                print(f"\n✓ Good digit - normal example")
            else:
                print(f"\n⚠️ Acceptable but somewhat atypical")
        
        print("="*60)
        
    except FileNotFoundError:
        print(f"\nError: Image file '{image_path}' not found!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
