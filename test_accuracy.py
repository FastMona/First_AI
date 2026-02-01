"""Automated accuracy/OOD smoke test.

Why: exercises the same detection_utils path across CNN/ART/FFN to surface
regressions early and mirror dashboard Option 4 without bespoke logic.
"""

from pathlib import Path
import os
from detection_utils import load_models, predict_image, parse_filename
from config import Config

def test_single_model(model_name, model_type_override=None):
    """Test a single model and return results"""
    print(f"\n{'='*80}")
    print(f"Testing {model_name} Model")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    clf, autoencoder, ood_detector, ae_threshold, model_type = load_models(model_type=model_type_override)
    
    if clf is None:
        print(f"\n⚠️  {model_name} model not trained yet. Skipping.")
        return None
    
    print(f"✓ All models loaded successfully (Using {model_type.upper()} classifier)\n")
    
    # Get test images
    folder_path = "test_images"
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return None
    
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
    
    # CONFUSION MATRIX
    print(f"\n{'CONFUSION MATRIX (OOD Detection Context):':<40}")
    print("┌─────────────────────────────┬──────────────────┬──────────────────┐")
    print("│                             │  Predicted: DIGIT│ Predicted: OOD   │")
    print("├─────────────────────────────┼──────────────────┼──────────────────┤")
    
    # Row 1: Actually Digit
    tp = correct_digits + incorrect_digits  # All digits accepted (whether correct class or not)
    fn = rejected_digits  # Digits rejected
    print(f"│ Actually: DIGIT             │  {tp:3d} (TP+FP)     │  {fn:3d} (FN)        │")
    
    # Row 2: Actually OOD
    fp = false_acceptances  # OOD accepted as digit
    tn = correct_rejections  # OOD correctly rejected
    print(f"│ Actually: OOD (non-digit)   │  {fp:3d} (FP)        │  {tn:3d} (TN)        │")
    print("└─────────────────────────────┴──────────────────┴──────────────────┘")
    
    # Metrics
    print(f"\n{'PERFORMANCE METRICS:':<40}")
    
    # For digit classification (of accepted digits)
    if digit_samples:
        digit_accuracy = correct_digits / len(digit_samples) * 100
        print(f"  Digit Classification (of accepted):")
        print(f"    ✓ Correct class: {correct_digits}/{len(digit_samples)} ({digit_accuracy:.1f}%)")
        print(f"    ✗ Wrong class:   {incorrect_digits}/{len(digit_samples)} ({incorrect_digits/len(digit_samples)*100:.1f}%)")
        print(f"    ✗ Rejected (FN): {rejected_digits}/{len(digit_samples)} ({rejected_digits/len(digit_samples)*100:.1f}%)")
    
    # For OOD detection
    if ood_samples:
        ood_accuracy = correct_rejections / len(ood_samples) * 100
        print(f"\n  OOD Detection:")
        print(f"    ✓ True Negative (TN):  {correct_rejections}/{len(ood_samples)} ({ood_accuracy:.1f}%)")
        print(f"    ✗ False Positive (FP): {false_acceptances}/{len(ood_samples)} ({false_acceptances/len(ood_samples)*100:.1f}%)")
    
    # Overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  Overall OOD Detection Metrics:")
    print(f"    Precision: {precision:.3f} (of predicted digits, how many are real digits)")
    print(f"    Recall:    {recall:.3f} (of real digits, how many were accepted)")
    print(f"    F1-Score:  {f1_score:.3f}")
    
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
        verdict = "✓ EXCELLENT PERFORMANCE!"
    elif overall_accuracy >= 75:
        verdict = "✓ GOOD PERFORMANCE"
    elif overall_accuracy >= 60:
        verdict = "⚠ MODERATE PERFORMANCE - Room for improvement"
    else:
        verdict = "✗ POOR PERFORMANCE - Needs tuning"
    
    print(verdict)
    print("="*80)
    
    # Return summary for comparison
    return {
        'model_name': model_name,
        'total_samples': total_samples,
        'correct_digits': correct_digits,
        'total_digits': len(digit_samples),
        'correct_rejections': correct_rejections,
        'total_ood': len(ood_samples),
        'overall_accuracy': overall_accuracy,
        'digit_accuracy': digit_accuracy if digit_samples else 0,
        'ood_accuracy': ood_accuracy if ood_samples else 0,
        'verdict': verdict
    }

def main():
    """Main function - tests all available models: CNN, ART, and FFN"""
    print("="*80)
    print("AUTOMATED ACCURACY TEST - MNIST Digit Detector")
    print("="*80)
    
    # Check which models are available
    cnn_available = os.path.exists(Config.MODEL_PATH_CNN)
    art_available = os.path.exists(Config.MODEL_PATH_ART)
    ffn_available = os.path.exists(Config.MODEL_PATH_FFN)
    
    if not cnn_available and not art_available and not ffn_available:
        print("\n⚠️  No trained models found!")
        print("\nPlease train a model first:")
        print("  - Option 1: Train with CNN")
        print("  - Option 2: Train with ART")
        print("  - Option 3: Train with FFN")
        return
    
    results = []
    
    # Test FFN if available
    if ffn_available:
        result = test_single_model("FFN", model_type_override='ffn')
        if result:
            results.append(result)
    else:
        print("\n⚠️  FFN model (model_state_ffn.pth) not found. Skipping FFN test.")
    
    # Test CNN if available
    if cnn_available:
        result = test_single_model("CNN", model_type_override='cnn')
        if result:
            results.append(result)
    else:
        print("\n⚠️  CNN model (model_state.pth) not found. Skipping CNN test.")
    
    # Test ART if available
    if art_available:
        result = test_single_model("ART", model_type_override='art')
        if result:
            results.append(result)
    else:
        print("\n⚠️  ART model (model_state_art.pth) not found. Skipping ART test.")
    
    # Comparison summary if multiple models were tested
    if len(results) >= 2:
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Create header
        header_parts = ['Metric']
        for r in results:
            header_parts.append(r['model_name'])
        
        print(f"\n{header_parts[0]:<22}", end='')
        for i in range(1, len(header_parts)):
            print(f"{header_parts[i]:>19}", end='')
        print()
        print("-"*80)
        
        # Overall accuracy
        print(f"{'Overall Accuracy:':<22}", end='')
        for r in results:
            print(f"{r['overall_accuracy']:>18.1f}%", end='')
        print()
        
        # Digit classification
        print(f"{'Digit Classification:':<22}", end='')
        for r in results:
            print(f"{r['digit_accuracy']:>18.1f}%", end='')
        print()
        
        # OOD detection
        print(f"{'OOD Detection:':<22}", end='')
        for r in results:
            print(f"{r['ood_accuracy']:>18.1f}%", end='')
        print()
        
        print("\n" + "-"*80)
        
        # Determine winner
        best_model = max(results, key=lambda x: x['overall_accuracy'])
        best_accuracy = best_model['overall_accuracy']
        
        # Check if there's a tie
        winners = [r for r in results if r['overall_accuracy'] == best_accuracy]
        
        if len(winners) == 1:
            print(f"\n🏆 Winner: {best_model['model_name']} (best overall accuracy: {best_accuracy:.1f}%)")
        else:
            winner_names = ', '.join([w['model_name'] for w in winners])
            print(f"\n🤝 Tie: {winner_names} (tied at {best_accuracy:.1f}%)")
        
        print("="*80)

if __name__ == "__main__":
    main()
