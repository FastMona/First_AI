# Shared utilities for MNIST digit detection
# Contains common functions used across detect.py, detect_batch.py, and test_accuracy.py

import torch
import re
from torch import load
from PIL import Image
from torchvision.transforms import ToTensor
from nn_model import ImageClassifier
from autoencoder_model import MNISTAutoencoder
from ood_detector import MahalanobisOODDetector

def load_models():
    """
    Load all required models for digit detection.
    
    Returns:
        tuple: (classifier, autoencoder, ood_detector, ae_threshold)
        Returns (None, None, None, None) if any model fails to load
    """
    try:
        # Load classifier
        clf = ImageClassifier().to('cuda')
        with open('model_state.pth', 'rb') as f:
            clf.load_state_dict(load(f))
        clf.eval()
        
        # Load autoencoder
        with open('autoencoder.pth', 'rb') as f:
            ae_data = load(f)
        autoencoder = MNISTAutoencoder(latent_dim=64).to('cuda')
        autoencoder.load_state_dict(ae_data['model_state'])
        autoencoder.eval()
        ae_threshold = ae_data['threshold_95']
        
        # Load OOD detector
        ood_detector = MahalanobisOODDetector('ood_params.pth')
        
        return clf, autoencoder, ood_detector, ae_threshold
        
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please train the models first using nn_train.py")
        return None, None, None, None

def predict_image(image_path, model, autoencoder, ood_detector, ae_threshold):
    """
    Predict digit with two-stage OOD detection.
    
    Args:
        image_path: Path to image file
        model: Trained classifier
        autoencoder: Trained autoencoder
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
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        # Stage 1: Reconstruction error
        recon_error = autoencoder.reconstruction_error(img_tensor).item()
        
        if recon_error > ae_threshold:
            # Rejected by autoencoder
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            prediction = torch.argmax(probs).item()
            confidence = probs[prediction].item()
            return prediction, confidence, False, recon_error, None, "reconstruction"
        
        # Stage 2: Mahalanobis distance
        features = model.get_features(img_tensor)
        output = model(img_tensor)
        
        probs = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probs).item()
        confidence = probs[prediction].item()
        
        belongs, mahal_distance, min_distance, nearest_class, all_distances = ood_detector.detect(
            features[0], prediction
        )
        
        if not belongs:
            return prediction, confidence, False, recon_error, min_distance, "mahalanobis"
    
    return prediction, confidence, True, recon_error, min_distance, "passed"

def parse_filename(filename):
    """
    Parse filename to extract ground truth label.
    
    Format: img_X.jpg where X is either:
    - A single digit (0-9): indicates the true digit label
    - Anything else: indicates OOD (not a digit)
    
    Args:
        filename: Image filename
    
    Returns:
        tuple: (is_digit, label)
        - is_digit: True if filename represents a digit, False if OOD
        - label: Digit value (0-9) if is_digit, None otherwise
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
