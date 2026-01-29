"""Batch digit detection entrypoint.

Why: reuses the same two-stage OOD path as single-image detection to surface
accuracy/acceptance stats across a folder, matching dashboard Option 6 and
preventing drift from detection_utils behavior.
"""

import logging
import os
from pathlib import Path
from detection_utils import load_models, predict_image
from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main(folder_path=None):
    """Process all images in a folder for digit detection
    
    Args:
        folder_path (str): Path to folder containing images. If None, will prompt user or use default.
    """
    print("="*80)
    print("MNIST Batch Digit Detector with 2-Stage OOD Detection")
    print("="*80)
    
    # Check which models are available
    cnn_available = os.path.exists(Config.MODEL_PATH)
    art_available = os.path.exists(Config.MODEL_PATH_ART)
    ffn_available = os.path.exists(Config.MODEL_PATH_FFN)
    
    print("\nAvailable trained models:")
    print(f"  1. FFN  {'✓ Trained' if ffn_available else '✗ Not trained'}")
    print(f"  2. CNN  {'✓ Trained' if cnn_available else '✗ Not trained'}")
    print(f"  3. ART  {'✓ Trained' if art_available else '✗ Not trained'}")
    
    if not (cnn_available or art_available or ffn_available):
        logger.error("No trained models found!")
        logger.info("Please train at least one model first:")
        logger.info("  - For CNN: Run 'python nn_train_cnn.py'")
        logger.info("  - For ART: Run 'python nn_train_art.py'")
        logger.info("  - For FFN: Run 'python nn_train_ffn.py'")
        return
    
    # Ask user to select model
    while True:
        choice = input("\nSelect model to test (1=FFN, 2=CNN, 3=ART): ").strip()
        
        if choice == '1':
            if not ffn_available:
                print("❌ FFN model not trained. Please run 'python nn_train_ffn.py' first.")
                continue
            model_type = 'ffn'
            break
        elif choice == '2':
            if not cnn_available:
                print("❌ CNN model not trained. Please run 'python nn_train_cnn.py' first.")
                continue
            model_type = 'cnn'
            break
        elif choice == '3':
            if not art_available:
                print("❌ ART model not trained. Please run 'python nn_train_art.py' first.")
                continue
            model_type = 'art'
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    # Load models (no need to print "Loading..." here, load_models() will do it)
    clf, autoencoder, ood_detector, ae_threshold, model_type = load_models(model_type)
    
    if clf is None:
        return
    
    logger.info(f"✓ All models loaded successfully! (Using {model_type.upper()} classifier)")
    
    # Get folder path
    if folder_path is None:
        folder_path = input("\nEnter folder path (default: test_images): ").strip()
        if not folder_path:
            folder_path = "test_images"
    else:
        print(f"\nProcessing folder: {folder_path}")
    
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
        except ValueError as e:
            # Feature dimension mismatch - this is a critical error that affects all images
            print(f"\n❌ ERROR: {e}")
            return
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
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
    
    # Final summary statistics
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    # Extract true labels from filenames if they follow pattern like "img_0.jpg", "img_1.jpg", etc.
    labeled_results = []
    for r in results:
        # Try to extract digit from filename (e.g., "img_3.jpg" -> 3, "test_5.png" -> 5)
        filename_lower = r['filename'].lower().replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        
        # Look for single digit in filename
        true_label = None
        for char in filename_lower:
            if char.isdigit():
                true_label = int(char)
                break
        
        if true_label is not None:
            labeled_results.append({
                'true_label': true_label,
                'prediction': r['prediction'],
                'belongs': r['belongs'],
                'filename': r['filename']
            })
    
    total_samples = len(results)
    accepted_samples = in_distribution
    rejected_samples = out_of_distribution
    
    print(f"\nTotal samples processed: {total_samples}")
    print(f"  ✓ Accepted as digits: {accepted_samples} ({accepted_samples/total_samples*100:.1f}%)")
    print(f"  ✗ Rejected as OOD: {rejected_samples} ({rejected_samples/total_samples*100:.1f}%)")
    
    if labeled_results:
        # Calculate accuracy on labeled samples
        correct_predictions = sum(1 for r in labeled_results 
                                 if r['belongs'] and r['prediction'] == r['true_label'])
        total_labeled = len(labeled_results)
        accuracy = correct_predictions / total_labeled * 100
        
        # Calculate rejection accuracy (correctly rejected non-digits)
        labeled_digits = [r for r in labeled_results if r['true_label'] in range(10)]
        if labeled_digits:
            correct_digit_accepts = sum(1 for r in labeled_digits if r['belongs'])
            digit_accuracy = correct_digit_accepts / len(labeled_digits) * 100
            
            print(f"\nAccuracy on labeled samples:")
            print(f"  Overall accuracy: {correct_predictions}/{total_labeled} ({accuracy:.1f}%)")
            print(f"  Digit acceptance rate: {correct_digit_accepts}/{len(labeled_digits)} ({digit_accuracy:.1f}%)")
            
            # Show misclassifications
            misclassified = [r for r in labeled_results 
                           if r['belongs'] and r['prediction'] != r['true_label']]
            if misclassified:
                print(f"\n  Misclassifications: {len(misclassified)}")
                for r in misclassified:
                    print(f"    {r['filename']}: predicted {r['prediction']}, actual {r['true_label']}")
    
    print("="*80)

if __name__ == "__main__":
    main()
