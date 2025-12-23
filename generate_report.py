# Generate markdown report with image thumbnails and results
# Creates a visual report of test accuracy

from pathlib import Path
from detection_utils import load_models, predict_image, parse_filename

def main():
    print("="*80)
    print("GENERATING MARKDOWN REPORT WITH IMAGE THUMBNAILS")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    clf, autoencoder, ood_detector, ae_threshold = load_models()
    
    if clf is None:
        return
    
    print("✓ All models loaded\n")
    
    # Get test images
    folder_path = Path("test_images")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    # Process all images
    results = []
    for img_path in sorted(image_files):
        is_digit, true_label = parse_filename(img_path.name)
        
        try:
            prediction, confidence, belongs, recon_error, distance, stage = predict_image(
                img_path, clf, autoencoder, ood_detector, ae_threshold
            )
            
            results.append({
                'filename': img_path.name,
                'path': str(img_path),
                'is_digit': is_digit,
                'true_label': true_label,
                'prediction': prediction,
                'confidence': confidence,
                'predicted_as_digit': belongs,
                'stage': stage
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    # Calculate statistics
    digit_samples = [r for r in results if r['is_digit']]
    ood_samples = [r for r in results if not r['is_digit']]
    
    correct_digits = sum(1 for r in digit_samples if r['predicted_as_digit'] and r['prediction'] == r['true_label'])
    correct_rejections = sum(1 for r in ood_samples if not r['predicted_as_digit'])
    total_correct = correct_digits + correct_rejections
    overall_accuracy = total_correct / len(results) * 100 if results else 0
    
    # Generate markdown report
    print("Generating markdown report...")
    
    md_content = []
    md_content.append("# MNIST Digit Detection Test Results")
    md_content.append("")
    md_content.append(f"**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append("")
    
    # Summary
    md_content.append("## Summary")
    md_content.append("")
    md_content.append(f"- **Total Images**: {len(results)}")
    md_content.append(f"- **Digit Samples**: {len(digit_samples)}")
    md_content.append(f"- **OOD Samples**: {len(ood_samples)}")
    md_content.append(f"- **Overall Accuracy**: {overall_accuracy:.1f}%")
    md_content.append("")
    
    if digit_samples:
        digit_accuracy = correct_digits / len(digit_samples) * 100
        md_content.append(f"- **Digit Classification Accuracy**: {digit_accuracy:.1f}% ({correct_digits}/{len(digit_samples)})")
    
    if ood_samples:
        ood_accuracy = correct_rejections / len(ood_samples) * 100
        md_content.append(f"- **OOD Detection Accuracy**: {ood_accuracy:.1f}% ({correct_rejections}/{len(ood_samples)})")
    
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    
    # Digit samples
    md_content.append("## Digit Samples (0-9)")
    md_content.append("")
    md_content.append("These images contain actual digits and should be classified correctly.")
    md_content.append("")
    
    if digit_samples:
        # Create table
        md_content.append("| Image | Filename | True Label | Prediction | Confidence | Result |")
        md_content.append("|-------|----------|------------|------------|------------|--------|")
        
        for r in digit_samples:
            img_md = f"![{r['filename']}]({r['path']})"
            
            if r['predicted_as_digit']:
                if r['prediction'] == r['true_label']:
                    result = "✅ CORRECT"
                    pred_str = f"{r['prediction']}"
                else:
                    result = f"❌ WRONG"
                    pred_str = f"{r['prediction']}"
            else:
                result = f"❌ REJECTED ({r['stage']})"
                pred_str = f"~~{r['prediction']}~~"
            
            conf_str = f"{r['confidence']*100:.1f}%"
            
            md_content.append(f"| {img_md} | `{r['filename']}` | **{r['true_label']}** | {pred_str} | {conf_str} | {result} |")
    
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    
    # OOD samples
    md_content.append("## Out-of-Distribution Samples")
    md_content.append("")
    md_content.append("These images do NOT contain digits and should be rejected by the OOD detector.")
    md_content.append("")
    
    if ood_samples:
        # Create table
        md_content.append("| Image | Filename | Expected | Detection Result | Stage | Classifier Guess |")
        md_content.append("|-------|----------|----------|------------------|-------|------------------|")
        
        for r in ood_samples:
            img_md = f"![{r['filename']}]({r['path']})"
            
            if not r['predicted_as_digit']:
                result = "✅ REJECTED"
                stage_str = r['stage']
            else:
                result = f"❌ ACCEPTED"
                stage_str = "passed"
            
            guess_str = f"{r['prediction']} ({r['confidence']*100:.1f}%)"
            
            md_content.append(f"| {img_md} | `{r['filename']}` | OOD | {result} | {stage_str} | {guess_str} |")
    
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    
    # Detection stages breakdown
    md_content.append("## Detection Stages Breakdown")
    md_content.append("")
    
    stage1_rejects = [r for r in results if not r['predicted_as_digit'] and r['stage'] == 'reconstruction']
    stage2_rejects = [r for r in results if not r['predicted_as_digit'] and r['stage'] == 'mahalanobis']
    passed = [r for r in results if r['predicted_as_digit']]
    
    md_content.append(f"- **Stage 1 (Autoencoder) Rejections**: {len(stage1_rejects)}")
    md_content.append(f"- **Stage 2 (Mahalanobis) Rejections**: {len(stage2_rejects)}")
    md_content.append(f"- **Passed Both Stages**: {len(passed)}")
    md_content.append("")
    
    # Footer
    md_content.append("---")
    md_content.append("")
    md_content.append("*Report generated by test_accuracy.py*")
    
    # Write to file
    output_file = "test_results_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    print(f"\n✅ Markdown report generated: {output_file}")
    print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
    print("="*80)

if __name__ == "__main__":
    main()
