"""Test and visualize class-conditional thresholds.

Shows threshold differences across digit classes and tests OOD rejection
behavior with noisy samples and non-digit images.

Dashboard Menu: Not directly called by dashboard (standalone testing utility)
"""

import torch
import numpy as np
import logging
from detection_utils import load_models, predict_image
from torch import load
from config import Config
import os
from PIL import Image

logger = logging.getLogger(__name__)

def show_threshold_info():
    """Display class-conditional threshold information"""
    logger.info("="*70)
    logger.info("CLASS-CONDITIONAL THRESHOLD ANALYSIS")
    logger.info("="*70)
    
    # Load OOD parameters
    with open(Config.OOD_PARAMS_PATH, 'rb') as f:
        params = load(f, weights_only=False)
    
    if 'class_thresholds_95' not in params:
        logger.error("\n❌ No class-conditional thresholds found!")
        logger.error("   Please retrain with: python nn_train.py")
        return None
    
    class_thresholds = params['class_thresholds_95']
    class_means = params.get('class_mean_distances', {})
    class_stds = params.get('class_std_distances', {})
    global_threshold = params.get('threshold_95', 'N/A')
    
    logger.info("\n📊 Per-Class Thresholds (95th percentile):")
    logger.info("-"*70)
    logger.info(f"{'Class':<8} {'Mean Dist':<12} {'Std Dev':<12} {'Threshold':<12} {'vs Global':<12}")
    logger.info("-"*70)
    
    thresholds_list = []
    for i in range(10):
        if i in class_thresholds:
            mean = class_means.get(i, 0)
            std = class_stds.get(i, 0)
            thresh = class_thresholds[i]
            diff = thresh - global_threshold if isinstance(global_threshold, float) else 0
            diff_str = f"{diff:+.2f}" if isinstance(global_threshold, float) else "N/A"
            logger.info(f"  {i:<6} {mean:>8.2f}     {std:>8.2f}     {thresh:>8.2f}     {diff_str:>8}")
            thresholds_list.append(thresh)
    
    logger.info("-"*70)
    if isinstance(global_threshold, float):
        logger.info(f"Global threshold (reference): {global_threshold:.2f}")
    logger.info(f"\nThreshold range: [{min(thresholds_list):.2f}, {max(thresholds_list):.2f}]")
    logger.info(f"Variation: {max(thresholds_list) - min(thresholds_list):.2f}")
    
    # Identify most/least strict classes
    most_strict = min(range(10), key=lambda i: class_thresholds[i])
    most_lenient = max(range(10), key=lambda i: class_thresholds[i])
    
    logger.info(f"\n🔒 Most strict: Class {most_strict} (threshold: {class_thresholds[most_strict]:.2f})")
    logger.info(f"🔓 Most lenient: Class {most_lenient} (threshold: {class_thresholds[most_lenient]:.2f})")
    
    return class_thresholds

def test_with_noise():
    """Test rejection with artificially noisy images"""
    logger.info("\n" + "="*70)
    logger.info("TESTING OOD REJECTION WITH NOISY MNIST SAMPLES")
    logger.info("="*70)
    
    # Load models
    clf, autoencoder, ood_detector, ae_threshold = load_models()
    if clf is None:
        return
    
    # Find a test image
    test_images_dir = 'test_images'
    if not os.path.exists(test_images_dir):
        logger.error(f"\n❌ Directory '{test_images_dir}' not found!")
        return
    
    test_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not test_files:
        logger.error(f"\n❌ No images found in '{test_images_dir}'!")
        return
    
    test_image = os.path.join(test_images_dir, test_files[0])
    logger.info(f"\n📷 Base image: {test_files[0]}")
    
    # Test original
    from torchvision.transforms import ToTensor
    img = Image.open(test_image)
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        output = clf(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probs).item()
        
        features = clf.get_features(img_tensor)
        predicted_label = torch.tensor([prediction], dtype=torch.long, device='cuda')
        recon_error = autoencoder.reconstruction_error(img_tensor, predicted_label).item()
        
        belongs, distance, min_dist, nearest, all_dists = ood_detector.detect(features[0], prediction)
    
    # Get threshold
    if ood_detector.class_thresholds_95 and prediction in ood_detector.class_thresholds_95:
        threshold = ood_detector.class_thresholds_95[prediction]
    else:
        threshold = ood_detector.threshold_95
    
    logger.info(f"\n✓ Original image (class {prediction}):")
    logger.info(f"  Mahalanobis distance: {min_dist:.2f}")
    logger.info(f"  Class-{prediction} threshold: {threshold:.2f}")
    logger.info(f"  Margin: {threshold - min_dist:.2f} (belongs: {belongs})")
    
    # Test with increasing noise
    logger.info(f"\n🔬 Testing with added noise:")
    logger.info("-"*70)
    noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    for noise_std in noise_levels:
        # Add Gaussian noise
        noisy_tensor = img_tensor + torch.randn_like(img_tensor) * noise_std
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        
        with torch.no_grad():
            output = clf(noisy_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(probs).item()
            conf = probs[pred].item()
            
            features = clf.get_features(noisy_tensor)
            pred_label = torch.tensor([pred], dtype=torch.long, device='cuda')
            recon_err = autoencoder.reconstruction_error(noisy_tensor, pred_label).item()
            
            belongs, dist, min_d, nearest, all_d = ood_detector.detect(features[0], pred)
            
            # Get threshold for this prediction
            if ood_detector.class_thresholds_95 and pred in ood_detector.class_thresholds_95:
                thresh = ood_detector.class_thresholds_95[pred]
            else:
                thresh = ood_detector.threshold_95
        
        status = "✓ ACCEPT" if belongs else "✗ REJECT"
        recon_status = "✓" if recon_err <= ae_threshold else "✗"
        mahal_status = "✓" if min_d < thresh else "✗"
        
        logger.info(f"  Noise σ={noise_std:4.1f}: pred={pred} conf={conf:.2f} | "
              f"Recon={recon_err:.4f}{recon_status} Mahal={min_d:5.2f}{mahal_status} (thresh={thresh:.2f}) | {status}")

def test_alphabet_images():
    """Test with non-digit images (alphabet)"""
    logger.info("\n" + "="*70)
    logger.info("TESTING OOD REJECTION WITH NON-DIGIT IMAGES")
    logger.info("="*70)
    
    alphabet_dir = 'alphabet_images'
    if not os.path.exists(alphabet_dir):
        logger.error(f"\n❌ Directory '{alphabet_dir}' not found!")
        logger.error("   Create some letter images to test OOD rejection")
        return
    
    # Load models
    clf, autoencoder, ood_detector, ae_threshold = load_models()
    if clf is None:
        return
    
    alphabet_files = [f for f in os.listdir(alphabet_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not alphabet_files:
        logger.error(f"\n❌ No images found in '{alphabet_dir}'!")
        return
    
    logger.info(f"\n📝 Testing {len(alphabet_files)} alphabet images:")
    logger.info("-"*70)
    
    accepted = 0
    rejected_stage1 = 0
    rejected_stage2 = 0
    
    for img_file in sorted(alphabet_files)[:10]:  # Test up to 10 images
        img_path = os.path.join(alphabet_dir, img_file)
        try:
            pred, conf, belongs, recon_err, dist, stage = predict_image(
                img_path, clf, autoencoder, ood_detector, ae_threshold
            )
            
            # Get threshold
            if ood_detector.class_thresholds_95 and pred in ood_detector.class_thresholds_95:
                thresh = ood_detector.class_thresholds_95[pred]
            else:
                thresh = ood_detector.threshold_95
            
            if belongs:
                status = "✓ ACCEPT"
                accepted += 1
            elif stage == "reconstruction":
                status = "✗ REJECT (Stage 1)"
                rejected_stage1 += 1
            else:
                status = f"✗ REJECT (Stage 2, dist={dist:.2f} > {thresh:.2f})"
                rejected_stage2 += 1
            
            logger.info(f"  {img_file:<25} pred={pred} conf={conf:.2f} | {status}")
        except Exception as e:
            logger.error(f"  {img_file:<25} Error: {e}")
    
    logger.info("-"*70)
    logger.info(f"Summary: {accepted} accepted, {rejected_stage1} rejected at stage 1, {rejected_stage2} rejected at stage 2")
    
    if accepted > 0 and rejected_stage2 == 0:
        logger.warning("\n⚠️  No Stage 2 (Mahalanobis) rejections!")
        logger.warning("    The thresholds may be too lenient, or alphabet images pass Stage 1 rarely.")

def main():
    """Run all diagnostic tests"""
    logger.info("\n" + "="*70)
    logger.info("CLASS-CONDITIONAL THRESHOLD DIAGNOSTIC TOOL")
    logger.info("="*70)
    
    # 1. Show threshold information
    thresholds = show_threshold_info()
    if thresholds is None:
        return
    
    # 2. Test with noisy versions of valid digits
    test_with_noise()
    
    # 3. Test with alphabet images
    test_alphabet_images()
    
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS:")
    logger.info("="*70)
    logger.info("1. If you see no Stage 2 rejections:")
    logger.info("   - Thresholds may be too high (very permissive)")
    logger.info("   - Try testing with more challenging OOD samples")
    logger.info("   - Consider using threshold_99 or custom percentile")
    logger.info("\n2. To make thresholds stricter:")
    logger.info("   - Modify nn_train.py to use 90th or 85th percentile")
    logger.info("   - Edit: np.percentile(distances, 90) instead of 95")
    logger.info("\n3. To verify class-conditional behavior:")
    logger.info("   - Look for different thresholds per class (shown above)")
    logger.info("   - Test images that are close to each class boundary")
    logger.info("="*70)

if __name__ == "__main__":
    main()
