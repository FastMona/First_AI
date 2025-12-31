# Shared utilities for MNIST digit detection
# Common functions for model loading, prediction, result formatting, and display
# Used by: detect.py, detect_batch.py, test_accuracy.py, generate_report.py

import torch
import re
from torch import load
from PIL import Image
from torchvision.transforms import ToTensor
from nn_model import ImageClassifier
from autoencoder_model import MNISTAutoencoder
from ood_detector import MahalanobisOODDetector
from config import Config

def load_models():
    """
    Load all required models for digit detection.
    
    Returns:
        tuple: (classifier, autoencoder, ood_detector, ae_threshold)
        Returns (None, None, None, None) if any model fails to load
    """
    try:
        # Load classifier
        clf = ImageClassifier().to(Config.DEVICE)
        with open(Config.MODEL_PATH, 'rb') as f:
            clf.load_state_dict(load(f, weights_only=False))
        clf.eval()
        
        # Load autoencoder
        with open(Config.AUTOENCODER_PATH, 'rb') as f:
            ae_data = load(f, weights_only=False)
        autoencoder = MNISTAutoencoder(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
        autoencoder.load_state_dict(ae_data['model_state'])
        autoencoder.eval()
        ae_threshold = ae_data['threshold_95']
        
        # Load OOD detector
        ood_detector = MahalanobisOODDetector(Config.OOD_PARAMS_PATH)
        
        return clf, autoencoder, ood_detector, ae_threshold
        
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please train the models first using nn_train.py")
        return None, None, None, None

def predict_image(image_path, model, autoencoder, ood_detector, ae_threshold):
    """
    Predict digit with two-stage OOD detection using class-conditional autoencoder.
    
    Two-stage biological perception model:
    1. Classifier predicts: "I think this is a 3"
    2. Stage 1 - Autoencoder checks: "Does it look like a 3?" (reconstruction)
    3. Stage 2 - Mahalanobis checks: "Is it close to the 3 prototype?" (distance)
    
    Args:
        image_path: Path to image file
        model: Trained classifier
        autoencoder: Trained class-conditional autoencoder
        ood_detector: MahalanobisOODDetector instance
        ae_threshold: Reconstruction error threshold
    
    Returns:
        tuple: (prediction, confidence, belongs, recon_error, distance, rejection_stage)
        - prediction: Predicted digit (0-9)
        - confidence: Classifier confidence
        - belongs: True if accepted as digit, False if rejected
        - recon_error: Reconstruction error value
        - distance: Mahalanobis distance (None if rejected at stage 1)
        - rejection_stage: 'reconstruction', 'mahalanobis', or 'passed'
    """
    img = Image.open(image_path)
    img_tensor = ToTensor()(img).unsqueeze(0).to(Config.DEVICE)
    
    with torch.no_grad():
        # First, get classifier prediction
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probs).item()
        confidence = probs[prediction].item()
        
        # Stage 1: Class-conditional reconstruction error
        # "I think this is a {prediction} — does it look like a {prediction}?"
        predicted_label = torch.tensor([prediction], dtype=torch.long, device=Config.DEVICE)
        recon_error = autoencoder.reconstruction_error(img_tensor, predicted_label).item()
        
        if recon_error > ae_threshold:
            # Stage 1 REJECTION: Image cannot be reconstructed as predicted digit
            # High reconstruction error indicates non-digit or highly corrupted digit
            return prediction, confidence, False, recon_error, None, "reconstruction"
        
        # Stage 2: Mahalanobis distance
        features = model.get_features(img_tensor)
        
        belongs, mahal_distance, min_distance, nearest_class, all_distances = ood_detector.detect(
            features[0], prediction
        )
        
        if not belongs:
            return prediction, confidence, False, recon_error, min_distance, "mahalanobis"
    
    return prediction, confidence, True, recon_error, min_distance, "passed"

def parse_filename(filename):
    """
    Parse filename to extract ground truth label for testing/validation.
    
    Expected filename format: img_X.ext where:
    - X is a single digit (0-9): indicates true digit label (for accuracy testing)
    - X is anything else: indicates OOD sample (non-digit, for rejection testing)
    
    Examples:
        'img_3.jpg' -> (True, 3)   # True digit, label 3
        'img_a.png' -> (False, None)  # OOD sample (letter)
        'img_cat.jpg' -> (False, None)  # OOD sample (text)
    
    Args:
        filename: Image filename (with or without path)
    
    Returns:
        tuple: (is_digit, label)
            - is_digit: True if filename represents a digit, False if OOD
            - label: Digit value (0-9) if is_digit is True, None otherwise
    """
    match = re.match(r'img_(.+)\.(jpg|jpeg|png|bmp|gif)', filename.lower())
    if not match:
        return False, None
    
    label_str = match.group(1)
    
    # Check if it's a single digit
    if label_str.isdigit() and len(label_str) == 1:
        return True, int(label_str)
    else:
        return False, None

def get_class_threshold(ood_detector, prediction):
    """
    Get the appropriate Mahalanobis distance threshold for a predicted class.
    
    Implements hierarchical threshold selection strategy:
    1. Class-specific 90th percentile (stricter, preferred)
    2. Class-specific 95th percentile (moderate fallback)
    3. Global 95th percentile (compatibility fallback)
    
    Args:
        ood_detector: MahalanobisOODDetector instance
        prediction: Predicted class label (0-9)
    
    Returns:
        tuple: (threshold, threshold_description)
            - threshold: Float value for Mahalanobis distance threshold
            - threshold_description: String describing threshold type (e.g., "class-3 90%")
    """
    if ood_detector.class_thresholds_90 and prediction in ood_detector.class_thresholds_90:
        return ood_detector.class_thresholds_90[prediction], f"class-{prediction} 90%"
    elif ood_detector.class_thresholds_95 and prediction in ood_detector.class_thresholds_95:
        return ood_detector.class_thresholds_95[prediction], f"class-{prediction} 95%"
    else:
        return ood_detector.threshold_95, "global"

def format_detection_result(prediction, confidence, belongs, recon_error, distance, 
                           stage, ood_detector, ae_threshold, verbose=True):
    """
    Format detection result with appropriate thresholds and human-readable messages.
    
    Provides unified formatting for detection results across all detection scripts.
    Can return either verbose (multi-line formatted string) or compact (dict) output.
    
    Args:
        prediction: Predicted digit (0-9)
        confidence: Classifier confidence probability (0-1)
        belongs: Whether sample is in-distribution (True) or OOD (False)
        recon_error: Reconstruction error value from autoencoder
        distance: Mahalanobis distance (None if rejected at stage 1)
        stage: Rejection stage ('reconstruction', 'mahalanobis', or 'passed')
        ood_detector: MahalanobisOODDetector instance for threshold lookup
        ae_threshold: Reconstruction error threshold from autoencoder
        verbose: If True, return formatted message string; if False, return info dict
    
    Returns:
        dict or str: Formatted result (dict if verbose=False, multi-line string if verbose=True)
    """
    mahal_threshold, threshold_type = get_class_threshold(ood_detector, prediction)
    
    result = {
        'prediction': prediction,
        'confidence': confidence,
        'belongs': belongs,
        'recon_error': recon_error,
        'distance': distance,
        'stage': stage,
        'ae_threshold': ae_threshold,
        'mahal_threshold': mahal_threshold,
        'threshold_type': threshold_type
    }
    
    if not verbose:
        return result
    
    # Build verbose message
    lines = []
    lines.append("=" * Config.SEPARATOR_WIDTH)
    
    if not belongs:
        if stage == "reconstruction":
            lines.append("❌ REJECTED AT STAGE 1: RECONSTRUCTION ERROR TOO HIGH")
            lines.append(f"\nThis image cannot be reconstructed as a digit.")
            lines.append(f"Reconstruction error: {recon_error:.6f} (threshold: {ae_threshold:.6f})")
            lines.append(f"\nClassifier's guess: {prediction} ({confidence*100:.1f}%)")
            lines.append("\n💡 Stage 1 Gate: Autoencoder REJECTED this as NOT a digit")
            lines.append("   The autoencoder learned only digits, so it can't recreate this.")
        else:
            lines.append("❌ REJECTED AT STAGE 2: MAHALANOBIS DISTANCE TOO HIGH")
            lines.append(f"\nReconstruction error: {recon_error:.6f} ✓ (passed stage 1)")
            lines.append(f"Mahalanobis distance: {distance:.2f} ✗ ({threshold_type} threshold: {mahal_threshold:.2f})")
            lines.append(f"\nClassifier's guess: {prediction} ({confidence*100:.1f}%)")
            lines.append("\n💡 Stage 1 passed, but Stage 2 Mahalanobis distance REJECTED")
            lines.append("   Image reconstructs OK but doesn't match digit prototypes.")
    else:
        lines.append(f"✓ PASSED BOTH STAGES - VALID DIGIT")
        lines.append(f"\n🔢 Predicted Digit: {prediction}")
        lines.append(f"   Confidence: {confidence*100:.1f}%")
        lines.append(f"\nStage 1 - Reconstruction error: {recon_error:.6f} ✓ (threshold: {ae_threshold:.6f})")
        lines.append(f"Stage 2 - Mahalanobis distance: {distance:.2f} ✓ ({threshold_type} threshold: {mahal_threshold:.2f})")
        
        relative_recon = recon_error / ae_threshold * 100
        relative_mahal = distance / mahal_threshold * 100
        
        if relative_recon < 50 and relative_mahal < 50:
            lines.append(f"\n💪 Excellent digit - very typical example!")
        elif relative_recon < 75 and relative_mahal < 75:
            lines.append(f"\n✓ Good digit - normal example")
        else:
            lines.append(f"\n⚠️ Acceptable but somewhat atypical")
    
    lines.append("=" * Config.SEPARATOR_WIDTH)
    
    return "\n".join(lines)

def print_separator(char='=', width=None):
    """Print a separator line"""
    width = width or Config.SEPARATOR_WIDTH
    print(char * width)

def print_header(text, char='=', width=None):
    """Print a centered header with separator lines"""
    width = width or Config.SEPARATOR_WIDTH
    print_separator(char, width)
    print(text.center(width))
    print_separator(char, width)
