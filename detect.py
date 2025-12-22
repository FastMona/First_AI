# Digit detection program for MNIST model
# Loads trained model and predicts digits from image files

import torch
import numpy as np
from PIL import Image
from torch import nn, load
from torchvision.transforms import ToTensor
from nn_model import ImageClassifier

def predict_image(image_path, model, confidence_threshold=0.80, ambiguity_threshold=0.15, entropy_threshold=0.8):
    """
    Predict the digit in an image file with uncertainty awareness and OOD detection
    
    Args:
        image_path: Path to the image file (should be 28x28 grayscale)
        model: Trained ImageClassifier model
        confidence_threshold: Minimum confidence to make a prediction (default: 0.80)
        ambiguity_threshold: Max difference between top-2 to flag ambiguity (default: 0.15)
        entropy_threshold: Max entropy to consider valid digit (default: 0.8, more sensitive)
    
    Returns:
        prediction: Predicted digit (0-9) or None if uncertain
        confidence: Confidence percentage
        all_scores: All class probabilities
        is_certain: Whether the model is confident in its prediction
        is_ambiguous: Whether top 2 predictions are close
        is_not_digit: Whether input is likely NOT a digit (out-of-distribution)
        is_suspicious: Whether confidence is suspiciously high (overfitting indicator)
        entropy: Entropy of probability distribution
        top_2: List of (digit, probability) for top 2 predictions
    """
    # Load and convert image to tensor
    img = Image.open(image_path)
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        
        # Calculate confidence scores
        probs = torch.softmax(output, dim=1)[0]
        
        # Calculate entropy (measure of distribution uniformity)
        # High entropy = confused model = probabilities spread across many classes
        # Low entropy = confident model = probability concentrated on few classes
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # Calculate variance of probabilities
        # Rotated/weird images often have unusual probability distributions
        prob_variance = torch.var(probs).item()
        
        # Get top 2 predictions
        top2_values, top2_indices = torch.topk(probs, 2)
        prediction = top2_indices[0].item()
        confidence = top2_values[0].item()
        second_best = top2_indices[1].item()
        second_confidence = top2_values[1].item()
        
        # Count how many classes have >5% probability
        # Normal digits: 1-2 classes
        # Weird/rotated: 3+ classes with significant probability
        significant_classes = (probs > 0.05).sum().item()
        
        # Calculate uncertainty metrics
        is_certain = confidence >= confidence_threshold
        is_ambiguous = (confidence - second_confidence) <= ambiguity_threshold
        
        # Enhanced OOD detection combining multiple signals:
        # 1. Entropy: High means confusion across classes
        # 2. Significant classes: More classes active = unusual input
        # 3. Low variance with high confidence: Spurious feature match
        is_not_digit = (
            entropy > entropy_threshold or  # Confused model
            significant_classes >= 4 or     # Too many active classes
            (confidence > 0.95 and prob_variance < 0.008)  # Suspiciously focused
        )
        
        # Suspiciously high confidence (>97%) combined with low entropy
        # Can indicate overfitting to wrong features (like rotated shapes)
        is_suspicious = confidence > 0.97 and entropy < 0.15
        
        top_2 = [
            (prediction, confidence),
            (second_best, second_confidence)
        ]
    
    return prediction, confidence, probs, is_certain, is_ambiguous, is_not_digit, is_suspicious, entropy, top_2

def main():
    print("="*60)
    print("MNIST Digit Detector")
    print("="*60)
    
    # Load the trained model
    print("\nLoading trained model from model_state.pth...")
    clf = ImageClassifier().to('cuda')
    
    try:
        with open('model_state.pth', 'rb') as f:
            clf.load_state_dict(load(f))
        clf.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: model_state.pth not found!")
        print("Please train the model first using torchnn.py")
        return
    
    # Get image filename from user
    print("\n" + "-"*60)
    image_path = input("Enter image filename (e.g., img_1.jpg): ")
    
    try:
        # Make prediction with uncertainty awareness and OOD detection
        prediction, confidence, probs, is_certain, is_ambiguous, is_not_digit, is_suspicious, entropy, top_2 = predict_image(image_path, clf)
        
        # Display results with uncertainty feedback
        print("\n" + "="*60)
        
        if is_suspicious:
            print("⚠️  SUSPICIOUS - Confidence TOO HIGH")
            print(f"Prediction: {prediction} ({confidence*100:.1f}%)")
            print(f"Entropy: {entropy:.3f} (extremely low)")
            print(f"\n💡 WARNING: >98% confidence is rare even for clear digits.")
            print("   This might be a non-digit triggering spurious features.")
            print("   The model has NEVER seen 'not a digit' during training,")
            print("   so it's forced to pick something from 0-9.")
            print("\n   ❓ Is this really a handwritten digit (0-9)?")
        elif is_not_digit:
            print("❌ NOT A DIGIT")
            print(f"This doesn't look like any digit (0-9)")
            print(f"Entropy: {entropy:.2f} (confusion across many classes)")
            print(f"\nModel's best guess would be: {prediction} ({confidence*100:.1f}%)")
            print("But probabilities are too spread out - likely NOT a number.")
            print("\n💡 This could be a letter, symbol, or unclear image.")
        elif not is_certain:
            print("⚠️  UNCERTAIN - CONFIDENCE TOO LOW")
            print(f"Best guess: {prediction} (only {confidence*100:.1f}% confident)")
            print(f"Entropy: {entropy:.2f}")
            print("\n💡 The model is NOT SURE about this prediction.")
            print("   The image may be unclear, rotated, or not a digit.")
        elif is_ambiguous:
            print(f"⚠️  AMBIGUOUS - Could be {top_2[0][0]} or {top_2[1][0]}")
            print(f"Most likely: {top_2[0][0]} ({top_2[0][1]*100:.1f}%)")
            print(f"But also could be: {top_2[1][0]} ({top_2[1][1]*100:.1f}%)")
            print(f"Difference: only {(top_2[0][1] - top_2[1][1])*100:.1f}%")
            print(f"Entropy: {entropy:.2f}")
            print("\n💡 The model sees features of both digits.")
        else:
            print(f"✓ CONFIDENT PREDICTION: {prediction}")
            print(f"   Confidence: {confidence*100:.1f}%")
            print(f"   Entropy: {entropy:.2f} (low = focused prediction)")
            if confidence > 0.95:
                print("   💪 Very high confidence - clear digit!")
        
        print("="*60)
        
        # Show uncertainty score and entropy explanation
        uncertainty = 1 - confidence
        print(f"\n📊 Uncertainty Score: {uncertainty*100:.1f}%")
        if uncertainty > 0.5:
            print("   (High uncertainty - model doesn't know)")
        elif uncertainty > 0.3:
            print("   (Moderate uncertainty - somewhat unsure)")
        else:
            print("   (Low uncertainty - model is confident)")
        
        print(f"\n📈 Entropy: {entropy:.3f}")
        print("   (Measures confusion: 0=certain, 2.3=maximum confusion)")
        if entropy > 0.8:
            print("   ⚠️ High entropy - NOT a clear digit!")
        elif entropy > 0.5:
            print("   Moderate entropy - some confusion")
        elif entropy < 0.15:
            print("   ⚠️ EXTREMELY low entropy - suspiciously confident!")
            print("   (Might be a non-digit matching learned features)")
        else:
            print("   ✓ Low entropy - clear, focused prediction")
        
        print("\n" + "-"*60)
        print("All class probabilities:")
        for digit in range(10):
            bar = "█" * int(probs[digit] * 50)
            marker = " ← TOP" if digit == top_2[0][0] else (" ← 2nd" if digit == top_2[1][0] else "")
            print(f"  {digit}: {probs[digit]*100:5.2f}% {bar}{marker}")
        print("-"*60)
        
    except FileNotFoundError:
        print(f"\nError: Image file '{image_path}' not found!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
