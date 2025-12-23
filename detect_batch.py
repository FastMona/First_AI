# Batch digit detection for MNIST model
# Processes all images in a folder and reports results

from pathlib import Path
from detection_utils import load_models, predict_image

def main():
    print("="*80)
    print("MNIST Batch Digit Detector with 2-Stage OOD Detection")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    clf, autoencoder, ood_detector, ae_threshold = load_models()
    
    if clf is None:
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
            prediction, confidence, belongs, recon_error, distance, stage = predict_image(
                img_path, clf, autoencoder, ood_detector, ae_threshold
            )
            results.append({
                'filename': img_path.name,
                'prediction': prediction,
                'confidence': confidence,
                'belongs': belongs,
                'recon_error': recon_error,
                'distance': distance,
                'rejection_stage': stage
            })
        except Exception as e:
            results.append({
                'filename': img_path.name,
                'prediction': None,
                'confidence': 0,
                'belongs': False,
                'recon_error': 999,
                'distance': 999,
                'rejection_stage': 'error',
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
    print(f"{'Filename':<20} {'Status':<20} {'Pred':<6} {'Conf':<8} {'Recon Err':<12} {'Stage':<15}")
    print("-"*80)
    
    for r in results:
        if 'error' in r:
            print(f"{r['filename']:<20} {'ERROR':<20} {'-':<6} {'-':<8} {'-':<12} {'-':<15}")
        elif r['belongs']:
            status = "✓ DIGIT"
            pred_str = f"{r['prediction']}"
            conf_str = f"{r['confidence']*100:.1f}%"
            recon_str = f"{r['recon_error']:.4f}"
            stage_str = "Passed both"
            print(f"{r['filename']:<20} {status:<20} {pred_str:<6} {conf_str:<8} {recon_str:<12} {stage_str:<15}")
        else:
            status = "❌ NOT DIGIT"
            pred_str = f"({r['prediction']})"
            conf_str = f"{r['confidence']*100:.1f}%"
            recon_str = f"{r['recon_error']:.4f}"
            stage_str = f"Reject: {r['rejection_stage']}"
            print(f"{r['filename']:<20} {status:<20} {pred_str:<6} {conf_str:<8} {recon_str:<12} {stage_str:<15}")
    
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
        print("\nThese samples were REJECTED:")
        
        stage1_rejects = [r for r in ood_results if r['rejection_stage'] == 'reconstruction']
        stage2_rejects = [r for r in ood_results if r['rejection_stage'] == 'mahalanobis']
        
        if stage1_rejects:
            print(f"\nStage 1 Rejections (Autoencoder - {len(stage1_rejects)} samples):")
            for r in stage1_rejects:
                print(f"  - {r['filename']:<25} recon_error={r['recon_error']:.4f} (classifier guessed: {r['prediction']})")
        
        if stage2_rejects:
            print(f"\nStage 2 Rejections (Mahalanobis - {len(stage2_rejects)} samples):")
            for r in stage2_rejects:
                print(f"  - {r['filename']:<25} distance={r['distance']:.2f} (classifier guessed: {r['prediction']})")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
