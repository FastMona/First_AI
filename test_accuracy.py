# Automated accuracy test for MNIST digit detector
# Tests against labeled images in test_images folder

from pathlib import Path
from detection_utils import load_models, predict_image, parse_filename

def main():
    print("="*80)
    print("AUTOMATED ACCURACY TEST - MNIST Digit Detector")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    clf, autoencoder, ood_detector, ae_threshold = load_models()
    
    if clf is None:
        return
    
    print("✓ All models loaded successfully\n")
    
    # Get test images
    folder_path = "test_images"
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Testing on {len(image_files)} images from {folder_path}...")
    print("="*80)
    
    # Test each image
    results = []
    for img_path in sorted(image_files):
        is_digit, true_label = parse_filename(img_path.name)
        
        try:
            prediction, confidence, belongs, recon_error, distance, stage = predict_image(
                img_path, clf, autoencoder, ood_detector, ae_threshold
            )
            
            results.append({
                'filename': img_path.name,
                'is_digit': is_digit,
                'true_label': true_label,
                'prediction': prediction,
                'confidence': confidence,
                'predicted_as_digit': belongs,
                'stage': stage
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    # Calculate accuracy metrics
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    digit_samples = [r for r in results if r['is_digit']]
    ood_samples = [r for r in results if not r['is_digit']]
    
    # Digit classification accuracy
    correct_digits = 0
    incorrect_digits = 0
    rejected_digits = 0
    
    print("\nDIGIT SAMPLES (should be classified correctly):")
    print("-"*80)
    for r in digit_samples:
        if r['predicted_as_digit']:
            if r['prediction'] == r['true_label']:
                status = "✓ CORRECT"
                correct_digits += 1
            else:
                status = f"✗ WRONG (predicted {r['prediction']})"
                incorrect_digits += 1
        else:
            status = f"✗ REJECTED ({r['stage']})"
            rejected_digits += 1
        
        print(f"  {r['filename']:<20} True: {r['true_label']}  {status}")
    
    # OOD detection accuracy
    correct_rejections = 0
    false_acceptances = 0
    
    print("\nOOD SAMPLES (should be rejected):")
    print("-"*80)
    for r in ood_samples:
        if not r['predicted_as_digit']:
            status = f"✓ CORRECT REJECTION ({r['stage']})"
            correct_rejections += 1
        else:
            status = f"✗ FALSE ACCEPTANCE (predicted as {r['prediction']})"
            false_acceptances += 1
        
        print(f"  {r['filename']:<20} OOD     {status}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("ACCURACY SUMMARY")
    print("="*80)
    
    total_samples = len(results)
    total_correct = correct_digits + correct_rejections
    
    print(f"\nTotal samples: {total_samples}")
    print(f"  - Digit samples: {len(digit_samples)}")
    print(f"  - OOD samples: {len(ood_samples)}")
    
    print(f"\n{'DIGIT CLASSIFICATION PERFORMANCE:':<40}")
    if digit_samples:
        digit_accuracy = correct_digits / len(digit_samples) * 100
        print(f"  Correctly classified: {correct_digits}/{len(digit_samples)} ({digit_accuracy:.1f}%)")
        print(f"  Incorrectly classified: {incorrect_digits}/{len(digit_samples)} ({incorrect_digits/len(digit_samples)*100:.1f}%)")
        print(f"  Rejected (false negatives): {rejected_digits}/{len(digit_samples)} ({rejected_digits/len(digit_samples)*100:.1f}%)")
    
    print(f"\n{'OOD DETECTION PERFORMANCE:':<40}")
    if ood_samples:
        ood_accuracy = correct_rejections / len(ood_samples) * 100
        print(f"  Correctly rejected: {correct_rejections}/{len(ood_samples)} ({ood_accuracy:.1f}%)")
        print(f"  False acceptances: {false_acceptances}/{len(ood_samples)} ({false_acceptances/len(ood_samples)*100:.1f}%)")
    
    print(f"\n{'OVERALL ACCURACY:':<40}")
    overall_accuracy = total_correct / total_samples * 100
    print(f"  {total_correct}/{total_samples} correct ({overall_accuracy:.1f}%)")
    
    # Breakdown by stage
    stage1_rejects = len([r for r in results if not r['predicted_as_digit'] and r['stage'] == 'reconstruction'])
    stage2_rejects = len([r for r in results if not r['predicted_as_digit'] and r['stage'] == 'mahalanobis'])
    
    print(f"\n{'REJECTION BREAKDOWN:':<40}")
    print(f"  Stage 1 (Autoencoder) rejections: {stage1_rejects}")
    print(f"  Stage 2 (Mahalanobis) rejections: {stage2_rejects}")
    
    print("\n" + "="*80)
    
    # Final verdict
    if overall_accuracy >= 90:
        print("✓ EXCELLENT PERFORMANCE!")
    elif overall_accuracy >= 75:
        print("✓ GOOD PERFORMANCE")
    elif overall_accuracy >= 60:
        print("⚠ MODERATE PERFORMANCE - Room for improvement")
    else:
        print("✗ POOR PERFORMANCE - Needs tuning")
    
    print("="*80)

if __name__ == "__main__":
    main()
