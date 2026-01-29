"""Simple digit detection without OOD detection.

Why: Provides baseline NN-only predictions without the autoencoder or Mahalanobis
distance checks. Useful for comparing raw NN performance vs OOD-gated predictions.
"""

import logging
import sys
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from nn_model_cnn import ImageClassifier
from nn_model_art import FuzzyARTClassifier
from nn_model_ffn import FeedforwardClassifier
from config import Config
from pathlib import Path
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_available_models():
    """
    Check which models have been trained.
    
    Returns:
        dict: Dictionary with model types as keys and existence as boolean values
    """
    return {
        'cnn': os.path.exists(Config.MODEL_PATH),
        'art': os.path.exists(Config.MODEL_PATH_ART),
        'ffn': os.path.exists(Config.MODEL_PATH_FFN)
    }


def select_model_type():
    """
    Select which model to use. Prompts user if multiple are available.
    Only auto-selects if exactly one model exists.
    
    Returns:
        str: Selected model type ('cnn', 'art', or 'ffn'), or None if no models available
    """
    available = get_available_models()
    available_list = [model for model, exists in available.items() if exists]
    
    # No models available
    if not available_list:
        print("\n" + "="*80)
        print("  ❌ ERROR: NO TRAINED MODELS FOUND".center(80))
        print("="*80)
        print("\nNo trained neural network models were found!")
        print("\nPlease train a model first using one of these options:")
        print("  • python nn_train_cnn.py  - Train CNN model")
        print("  • python nn_train_art.py  - Train Fuzzy ART model")
        print("  • python nn_train_ffn.py  - Train Feedforward model")
        print("\nOr use the dashboard (python dashboard.py) and select:")
        print("  • Option 1 - Train with FFN")
        print("  • Option 2 - Train with CNN")
        print("  • Option 3 - Train with ART")
        print("="*80 + "\n")
        return None
    
    # Single model available - use it automatically (no prompting)
    if len(available_list) == 1:
        selected = available_list[0]
        print(f"✓ Using {selected.upper()} model")
        return selected
    
    # Multiple models available - prompt user to choose
    print("\n" + "─"*80)
    print("Multiple trained models found. Please select which one to use:")
    print("─"*80)
    
    model_names = {'cnn': 'CNN (Convolutional Neural Network)', 
                   'art': 'Fuzzy ART (Adaptive Resonance Theory)',
                   'ffn': 'FFN (Feedforward Neural Network)'}
    
    for i, model in enumerate(available_list, 1):
        print(f"  {i}. {model.upper():3} - {model_names[model]}")
    
    print("─"*80)
    
    while True:
        try:
            choice = input(f"\n➤ Select model (1-{len(available_list)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_list):
                selected = available_list[choice_idx]
                print(f"✓ Selected {selected.upper()} model")
                return selected
            else:
                print(f"❌ Please enter a number between 1 and {len(available_list)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n❌ Selection cancelled")
            return None


def load_classifier(model_type=None):
    """
    Load the trained classifier model.
    
    Args:
        model_type: 'cnn', 'art', or 'ffn'. If None, will prompt user to select.
    
    Returns:
        tuple: (classifier, model_type_used) or (None, None) if loading fails
    """
    try:
        # Select model type if not specified
        if model_type is None:
            model_type = select_model_type()
            if model_type is None:
                return None, None
        
        print(f"\nLoading {model_type.upper()} classifier...")
        
        # Load appropriate classifier
        if model_type == 'cnn':
            clf = ImageClassifier().to(Config.DEVICE)
            with open(Config.MODEL_PATH, 'rb') as f:
                clf.load_state_dict(torch.load(f, map_location=Config.DEVICE, weights_only=False))
        elif model_type == 'art':
            clf = FuzzyARTClassifier(
                input_dim=Config.INPUT_SIZE * Config.INPUT_SIZE,
                max_categories=Config.ART_MAX_CATEGORIES,
                vigilance=Config.ART_VIGILANCE,
                learning_rate=Config.ART_LEARNING_RATE,
                choice_alpha=Config.ART_CHOICE_ALPHA
            ).to(Config.DEVICE)
            with open(Config.MODEL_PATH_ART, 'rb') as f:
                clf.load_state_dict(torch.load(f, map_location=Config.DEVICE, weights_only=False))
        elif model_type == 'ffn':
            clf = FeedforwardClassifier(
                input_size=Config.INPUT_SIZE * Config.INPUT_SIZE,
                num_classes=Config.NUM_CLASSES,
                hidden_sizes=Config.FFN_HIDDEN_SIZES,
                embedding_size=Config.FEATURE_DIM
            ).to(Config.DEVICE)
            with open(Config.MODEL_PATH_FFN, 'rb') as f:
                clf.load_state_dict(torch.load(f, map_location=Config.DEVICE, weights_only=False))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        clf.eval()
        return clf, model_type
        
    except FileNotFoundError as e:
        print(f"Error loading classifier: {e}")
        print(f"Please train a model first using nn_train_cnn.py, nn_train_art.py, or nn_train_ffn.py")
        return None, None


def predict_simple(image_path, model):
    """
    Predict digit using only the neural network classifier.
    No OOD detection - just raw NN prediction.
    
    Args:
        image_path: Path to image file
        model: Trained classifier
    
    Returns:
        tuple: (prediction, confidence, probabilities)
        - prediction: Predicted digit (0-9)
        - confidence: Probability for predicted class
        - probabilities: Full probability distribution over all classes
    """
    img = Image.open(image_path)
    img_tensor = ToTensor()(img).unsqueeze(0).to(Config.DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probs).item()
        confidence = probs[prediction].item()
        
    return prediction, confidence, probs.cpu().numpy()


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}".center(80))
    print("="*80)


def format_result(prediction, confidence, probabilities, image_path):
    """Format the prediction result for display"""
    result = []
    result.append("\n" + "─"*80)
    result.append(f"Image: {image_path}")
    result.append("─"*80)
    result.append(f"\n🎯 PREDICTION: {prediction}")
    result.append(f"   Confidence: {confidence:.1%}")
    result.append("\n📊 Probability Distribution:")
    
    # Show all probabilities
    for digit in range(10):
        prob = probabilities[digit]
        bar_length = int(prob * 40)  # Scale to 40 characters max
        bar = '█' * bar_length
        result.append(f"   Digit {digit}: {prob:6.1%} {bar}")
    
    result.append("─"*80)
    return "\n".join(result)


def parse_filename(filename):
    """
    Parse filename to determine if it's a digit image and extract the true label.
    
    Expected format: img_0.jpg, img_1.png, etc. (for digits)
                    letter_A.jpg, random_xyz.png, etc. (for non-digits)
    
    Returns:
        tuple: (is_digit, true_label)
        - is_digit: True if filename indicates a digit image
        - true_label: The digit (0-9) if is_digit=True, else None
    """
    import re
    # Try to extract digit from filename like "img_0.jpg", "test_5.png", "digit_3.jpg"
    match = re.search(r'(?:img|test|digit|number)_(\d)(?:\D|$)', filename.lower())
    if match:
        return True, int(match.group(1))
    
    # If filename is just a digit with extension
    match = re.search(r'^(\d)\.', filename)
    if match:
        return True, int(match.group(1))
    
    # Otherwise assume it's not a digit
    return False, None


def test_on_dataset(model, model_type):
    """
    Test the model on test_images folder and generate confusion matrix.
    
    Args:
        model: Trained classifier
        model_type: Type of model ('cnn', 'art', 'ffn')
    """
    print("\n" + "="*80)
    print(f"  TESTING {model_type.upper()} MODEL ON TEST DATASET (NN ONLY)".center(80))
    print("="*80)
    
    # Get folder path from user
    folder_path = input("\nEnter folder path (default: test_images): ").strip()
    if not folder_path:
        folder_path = "test_images"
    
    if not os.path.exists(folder_path):
        print(f"\n❌ Folder '{folder_path}' not found!")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
    
    if not image_files:
        print(f"\n❌ No images found in {folder_path}!")
        return
    
    print(f"\nTesting on {len(image_files)} images from {folder_path}/")
    print("─"*80)
    
    # Separate digit and non-digit samples
    digit_samples = []
    ood_samples = []
    
    for img_path in sorted(image_files):
        is_digit, true_label = parse_filename(img_path.name)
        if is_digit:
            digit_samples.append((img_path, true_label))
        else:
            ood_samples.append(img_path)
    
    print(f"  - Digit images: {len(digit_samples)}")
    print(f"  - Non-digit images: {len(ood_samples)}")
    print("─"*80)
    
    # Test digits
    correct_class = 0
    wrong_class = 0
    
    print("\n📊 DIGIT CLASSIFICATION RESULTS:")
    for img_path, true_label in digit_samples:
        try:
            prediction, confidence, _ = predict_simple(str(img_path), model)
            
            if prediction == true_label:
                status = "✓"
                correct_class += 1
            else:
                status = "✗"
                wrong_class += 1
            
            print(f"  {status} {img_path.name:<25} True: {true_label}, Pred: {prediction} (conf: {confidence:.1%})")
        except Exception as e:
            print(f"  ✗ {img_path.name:<25} Error: {e}")
            wrong_class += 1
    
    # Test non-digits (they will all be "false positives" since NN has no OOD detection)
    non_digit_predictions = {}
    
    if ood_samples:
        print("\n📊 NON-DIGIT IMAGE PREDICTIONS:")
        for img_path in ood_samples:
            try:
                prediction, confidence, _ = predict_simple(str(img_path), model)
                non_digit_predictions[img_path.name] = (prediction, confidence)
                print(f"  • {img_path.name:<25} Predicted: {prediction} (conf: {confidence:.1%})")
            except Exception as e:
                print(f"  ✗ {img_path.name:<25} Error: {e}")
    
    # Calculate metrics
    total_samples = len(digit_samples) + len(ood_samples)
    
    # For confusion matrix in digit vs non-digit context:
    # TP = digits predicted as any digit (correct or wrong class)
    # FP = non-digits predicted as any digit (all of them, since no OOD rejection)
    # FN = 0 (no digits rejected, since no OOD detection)
    # TN = 0 (no non-digits rejected, since no OOD detection)
    
    tp = len(digit_samples)  # All digits accepted (no rejection mechanism)
    fp = len(ood_samples)     # All non-digits accepted as digits
    fn = 0                     # No false negatives (no rejection)
    tn = 0                     # No true negatives (no rejection)
    
    # Display confusion matrix
    print("\n" + "="*80)
    print("  CONFUSION MATRIX (Binary: Digit vs Non-Digit)".center(80))
    print("="*80)
    print("┌─────────────────────────────┬──────────────────┬──────────────────┐")
    print("│                             │  Predicted: DIGIT│ Predicted: OOD   │")
    print("├─────────────────────────────┼──────────────────┼──────────────────┤")
    print(f"│ Actually: DIGIT             │  {tp:3d} (TP)       │  {fn:3d} (FN)       │")
    print(f"│ Actually: OOD (non-digit)   │  {fp:3d} (FP)       │  {tn:3d} (TN)       │")
    print("└─────────────────────────────┴──────────────────┴──────────────────┘")
    
    print("\n" + "="*80)
    print("  PERFORMANCE METRICS".center(80))
    print("="*80)
    
    # Digit classification accuracy
    if digit_samples:
        digit_accuracy = correct_class / len(digit_samples) * 100
        print(f"\n📊 Digit Classification Accuracy:")
        print(f"    ✓ Correct class: {correct_class}/{len(digit_samples)} ({digit_accuracy:.1f}%)")
        print(f"    ✗ Wrong class:   {wrong_class}/{len(digit_samples)} ({wrong_class/len(digit_samples)*100:.1f}%)")
    
    # OOD detection (note: NN has no OOD detection, so all non-digits are FP)
    if ood_samples:
        print(f"\n⚠️  OOD Detection (No Rejection - NN Only):")
        print(f"    All {len(ood_samples)} non-digit images were classified as digits")
        print(f"    False Positive Rate: 100% (expected - no OOD detection)")
    
    # Overall metrics
    if total_samples > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # All digits accepted
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📈 Binary Classification Metrics (Digit vs Non-Digit):")
        print(f"    Precision: {precision:.3f} (of predictions, how many are real digits)")
        print(f"    Recall:    {recall:.3f} (of real digits, how many were predicted)")
        print(f"    F1-Score:  {f1_score:.3f}")
    
    print("\n" + "="*80)
    print("\nℹ️  NOTE: Without OOD detection, the NN accepts ALL images as digits.")
    print("   Compare this with Option 5 (Single Image Detection) to see the")
    print("   benefit of the 2-stage OOD detection (autoencoder + Mahalanobis).")
    print("="*80 + "\n")


def main(image_path=None):
    """Main entry point for simple digit detection - processes entire folders"""
    print_header("Simple MNIST Digit Detector (NN Only - No OOD Detection)")
    
    # Load classifier (will prompt for model selection if multiple exist)
    clf, model_type = load_classifier()
    
    if clf is None:
        return
    
    logger.info(f"✓ Classifier loaded successfully! (Using {model_type.upper()} model)")
    logger.info("ℹ️  Note: This uses ONLY the neural network - no OOD detection")
    
    # Test on dataset/folder
    test_on_dataset(clf, model_type)


if __name__ == "__main__":
    main()
