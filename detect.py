# Digit detection program for MNIST model
# Loads trained model and predicts digits from image files

import torch
import numpy as np
from PIL import Image
from torch import nn, load
from torchvision.transforms import ToTensor
from nn_model import ImageClassifier
from ood_detector import MahalanobisOODDetector

def predict_image(image_path, model, ood_detector, confidence_threshold=0.80):
    """
    Predict the digit in an image file with Mahalanobis distance OOD detection
    
    Args:
        image_path: Path to the image file (should be 28x28 grayscale)
        model: Trained ImageClassifier model
        ood_detector: MahalanobisOODDetector instance
        confidence_threshold: Minimum confidence to make a prediction
    
    Returns:
        prediction: Predicted digit (0-9) or None if OOD
        confidence: Confidence percentage
        belongs: True if sample belongs to training distribution
        mahal_distance: Mahalanobis distance to predicted class
        min_distance: Minimum distance to any class
        nearest_class: Nearest class by Mahalanobis distance
        all_probs: All class probabilities
    """
    # Load and convert image to tensor
    img = Image.open(image_path)
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    
    # Extract features and make prediction
    with torch.no_grad():
        features = model.get_features(img_tensor)
        output = model(img_tensor)
        
        # Calculate confidence scores
        probs = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probs).item()
        confidence = probs[prediction].item()
        
        # OOD detection using Mahalanobis distance
        belongs, mahal_distance, min_distance, nearest_class, all_distances = ood_detector.detect(
            features[0], prediction
        )
        
        # Get threshold for display
        threshold = ood_detector.threshold_95
    
    return prediction, confidence, belongs, mahal_distance, min_distance, nearest_class, probs, all_distances, threshold

def main():
    print("="*60)
    print("MNIST Digit Detector with Mahalanobis OOD Detection")
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
    
    # Load OOD detector
    try:
        ood_detector = MahalanobisOODDetector('ood_params.pth')
    except FileNotFoundError:
        print("Error: ood_params.pth not found!")
        print("Please train the model first to generate OOD parameters")
        return
    
    # Get image filename from user
    print("\n" + "-"*60)
    image_path = input("Enter image filename (e.g., img_1.jpg): ")
    
    try:
        # Make prediction with OOD detection
        prediction, confidence, belongs, mahal_distance, min_distance, nearest_class, probs, all_distances, threshold = predict_image(
            image_path, clf, ood_detector
        )
        
        # Display results
        print("\n" + "="*60)
        
        if not belongs:
            print("❌ OUT-OF-DISTRIBUTION (NOT A DIGIT)")
            print(f"\nThis sample does NOT belong to the training distribution.")
            print(f"Mahalanobis distance: {min_distance:.2f} (threshold: {threshold:.2f})")
            print(f"\nClassifier's guess: {prediction} ({confidence*100:.1f}%)")
            print(f"Nearest class: {nearest_class} (distance: {all_distances[nearest_class]:.2f})")
            print("\n💡 This is likely NOT a handwritten digit (0-9).")
            print("   Could be a letter, symbol, or unclear image.")
        else:
            print(f"✓ BELONGS TO TRAINING DISTRIBUTION")
            print(f"\n🔢 Predicted Digit: {prediction}")
            print(f"   Confidence: {confidence*100:.1f}%")
            print(f"   Mahalanobis distance: {mahal_distance:.2f} (threshold: {threshold:.2f})")
            
            # Calculate relative distance
            relative_dist = mahal_distance / threshold * 100
            
            if relative_dist < 50:
                print(f"   💪 Very typical example of digit {prediction}! ({relative_dist:.0f}% of threshold)")
            elif relative_dist < 75:
                print(f"   ✓ Normal example of digit {prediction} ({relative_dist:.0f}% of threshold)")
            else:
                print(f"   ⚠️ Somewhat atypical for digit {prediction} ({relative_dist:.0f}% of threshold)")
        
        print("="*60)
        
        # Show Mahalanobis distances to all classes
        print(f"\n📏 Mahalanobis Distances to Class Prototypes:")
        sorted_distances = sorted(all_distances.items(), key=lambda x: x[1])
        for digit, dist in sorted_distances:
            marker = " ← Predicted" if digit == prediction else (" ← Nearest" if digit == nearest_class else "")
            bar = "█" * max(1, int((15 - dist) * 2))
            print(f"  Class {digit}: {dist:6.2f} {bar}{marker}")
        
        print("\n📊 Classifier Probabilities:")
        for digit in range(10):
            bar = "█" * int(probs[digit] * 50)
            marker = " ← Predicted" if digit == prediction else ""
            print(f"  {digit}: {probs[digit]*100:5.2f}% {bar}{marker}")
        print("-"*60)
        
    except FileNotFoundError:
        print(f"\nError: Image file '{image_path}' not found!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
