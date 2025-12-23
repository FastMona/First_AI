# Batch digit detection for MNIST model
# Processes all images in a folder and reports results

import torch
import os
from pathlib import Path
from PIL import Image
from torch import load
from torchvision.transforms import ToTensor
from nn_model import ImageClassifier
from ood_detector import MahalanobisOODDetector

def predict_image(image_path, model, ood_detector):
    """Predict digit with OOD detection"""
    img = Image.open(image_path)
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        features = model.get_features(img_tensor)
        output = model(img_tensor)
        
        probs = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probs).item()
        confidence = probs[prediction].item()
        
        belongs, mahal_distance, min_distance, nearest_class, all_distances = ood_detector.detect(
            features[0], prediction
        )
    
    return prediction, confidence, belongs, min_distance

def main():
    print("="*80)
    print("MNIST Batch Digit Detector")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    clf = ImageClassifier().to('cuda')
    
    try:
        with open('model_state.pth', 'rb') as f:
            clf.load_state_dict(load(f))
        clf.eval()
        print("✓ Model loaded")
    except FileNotFoundError:
        print("Error: model_state.pth not found! Train the model first.")
        return
    
    # Load OOD detector
    try:
        ood_detector = MahalanobisOODDetector('ood_params.pth')
    except FileNotFoundError:
        print("Error: ood_params.pth not found! Train the model first.")
        return
    
    # Get folder path
    folder_path = input("\nEnter folder path (default: test_images): ").strip()
    if not folder_path:
        folder_path = "test_images"
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"\nProcessing {len(image_files)} images from {folder_path}...")
    print("-"*80)
    
    # Process all images
    results = []
    for img_path in sorted(image_files):
        try:
            prediction, confidence, belongs, distance = predict_image(img_path, clf, ood_detector)
            results.append({
                'filename': img_path.name,
                'prediction': prediction,
                'confidence': confidence,
                'belongs': belongs,
                'distance': distance
            })
        except Exception as e:
            results.append({
                'filename': img_path.name,
                'prediction': None,
                'confidence': 0,
                'belongs': False,
                'distance': 999,
                'error': str(e)
            })
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Count statistics
    in_distribution = sum(1 for r in results if r['belongs'])
    out_of_distribution = len(results) - in_distribution
    
    print(f"\nTotal images: {len(results)}")
    print(f"In-distribution (digits): {in_distribution}")
    print(f"Out-of-distribution (not digits): {out_of_distribution}")
    
    # Display detailed results
    print("\n" + "-"*80)
    print(f"{'Filename':<20} {'Status':<15} {'Prediction':<12} {'Confidence':<12} {'Distance':<10}")
    print("-"*80)
    
    for r in results:
        if 'error' in r:
            print(f"{r['filename']:<20} {'ERROR':<15} {'-':<12} {'-':<12} {'-':<10}")
        elif r['belongs']:
            status = "✓ DIGIT"
            pred_str = f"{r['prediction']}"
            conf_str = f"{r['confidence']*100:.1f}%"
            dist_str = f"{r['distance']:.1f}"
            print(f"{r['filename']:<20} {status:<15} {pred_str:<12} {conf_str:<12} {dist_str:<10}")
        else:
            status = "❌ NOT DIGIT"
            pred_str = f"({r['prediction']})"
            conf_str = f"{r['confidence']*100:.1f}%"
            dist_str = f"{r['distance']:.1f}"
            print(f"{r['filename']:<20} {status:<15} {pred_str:<12} {conf_str:<12} {dist_str:<10}")
    
    print("-"*80)
    
    # Group by prediction for digits
    print("\n" + "="*80)
    print("DIGITS DETECTED (In-Distribution Only)")
    print("="*80)
    
    digit_results = [r for r in results if r['belongs'] and r['prediction'] is not None]
    
    if digit_results:
        from collections import Counter
        predictions = Counter(r['prediction'] for r in digit_results)
        
        for digit in range(10):
            if digit in predictions:
                count = predictions[digit]
                files = [r['filename'] for r in digit_results if r['prediction'] == digit]
                print(f"\nDigit {digit}: {count} image(s)")
                for f in files:
                    result = next(r for r in digit_results if r['filename'] == f)
                    print(f"  - {f:<25} (confidence: {result['confidence']*100:.1f}%, distance: {result['distance']:.1f})")
    else:
        print("No valid digits detected")
    
    # Show OOD samples
    ood_results = [r for r in results if not r['belongs']]
    if ood_results:
        print("\n" + "="*80)
        print("OUT-OF-DISTRIBUTION SAMPLES")
        print("="*80)
        print("\nThese samples do NOT belong to the digit distribution:")
        for r in ood_results:
            print(f"  - {r['filename']:<25} (distance: {r['distance']:.1f}, classifier guessed: {r['prediction']})")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
