"""Test and visualize class-conditional thresholds.

Shows threshold differences across digit classes and tests OOD rejection
behavior with noisy samples and non-digit images.

Dashboard Menu: Not directly called by dashboard (standalone testing utility)
"""

import torch
import numpy as np
from detection_utils import load_models, predict_image
from torch import load
from config import Config
import os
from PIL import Image

def show_threshold_info():
    """Display class-conditional threshold information"""
    print("="*70)
    print("CLASS-CONDITIONAL THRESHOLD ANALYSIS")
    print("="*70)
    
    # Load OOD parameters
    with open(Config.OOD_PARAMS_PATH, 'rb') as f:
        params = load(f, weights_only=False)
    
    if 'class_thresholds_95' not in params:
        print("\n❌ No class-conditional thresholds found!")
        print("   Please retrain with: python nn_train.py")
        return None
    
    class_thresholds = params['class_thresholds_95']
    class_means = params.get('class_mean_distances', {})
    class_stds = params.get('class_std_distances', {})
    global_threshold = params.get('threshold_95', 'N/A')
    
    print("\n📊 Per-Class Thresholds (95th percentile):")
    print("-"*70)
    print(f"{'Class':<8} {'Mean Dist':<12} {'Std Dev':<12} {'Threshold':<12} {'vs Global':<12}")
    print("-"*70)
    
    thresholds_list = []
    for i in range(10):
        if i in class_thresholds:
            mean = class_means.get(i, 0)
            std = class_stds.get(i, 0)
            thresh = class_thresholds[i]
            diff = thresh - global_threshold if isinstance(global_threshold, float) else 0
            diff_str = f"{diff:+.2f}" if isinstance(global_threshold, float) else "N/A"
            print(f"  {i:<6} {mean:>8.2f}     {std:>8.2f}     {thresh:>8.2f}     {diff_str:>8}")
            thresholds_list.append(thresh)
    
    print("-"*70)
    if isinstance(global_threshold, float):
        print(f"Global threshold (reference): {global_threshold:.2f}")
    print(f"\nThreshold range: [{min(thresholds_list):.2f}, {max(thresholds_list):.2f}]")
    print(f"Variation: {max(thresholds_list) - min(thresholds_list):.2f}")
    
    # Identify most/least strict classes
    most_strict = min(range(10), key=lambda i: class_thresholds[i])
    most_lenient = max(range(10), key=lambda i: class_thresholds[i])
    
    print(f"\n🔒 Most strict: Class {most_strict} (threshold: {class_thresholds[most_strict]:.2f})")
    print(f"🔓 Most lenient: Class {most_lenient} (threshold: {class_thresholds[most_lenient]:.2f})")
    
    return class_thresholds

def test_with_noise():
    """Test rejection with artificially noisy images"""
    print("\n" + "="*70)
    print("TESTING OOD REJECTION WITH NOISY MNIST SAMPLES")
    print("="*70)
    
    # Load models
    clf, autoencoder, ood_detector, ae_threshold = load_models()
    if clf is None:
        return
    
    # Find a test image
    test_images_dir = 'test_images'
    if not os.path.exists(test_images_dir):
        print(f"\n❌ Directory '{test_images_dir}' not found!")
        return
    
    test_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not test_files:
        print(f"\n❌ No images found in '{test_images_dir}'!")
        return
    
    test_image = os.path.join(test_images_dir, test_files[0])
    print(f"\n📷 Base image: {test_files[0]}")
    
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
    
    print(f"\n✓ Original image (class {prediction}):")
    print(f"  Mahalanobis distance: {min_dist:.2f}")
    print(f"  Class-{prediction} threshold: {threshold:.2f}")
    print(f"  Margin: {threshold - min_dist:.2f} (belongs: {belongs})")
    
    # Test with increasing noise
    print(f"\n🔬 Testing with added noise:")
    print("-"*70)
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
        
        print(f"  Noise σ={noise_std:4.1f}: pred={pred} conf={conf:.2f} | "
              f"Recon={recon_err:.4f}{recon_status} Mahal={min_d:5.2f}{mahal_status} (thresh={thresh:.2f}) | {status}")

def test_alphabet_images():
    """Test with non-digit images (alphabet)"""
    print("\n" + "="*70)
    print("TESTING OOD REJECTION WITH NON-DIGIT IMAGES")
    print("="*70)
    
    alphabet_dir = 'alphabet_images'
    if not os.path.exists(alphabet_dir):
        print(f"\n❌ Directory '{alphabet_dir}' not found!")
        print("   Create some letter images to test OOD rejection")
        return
    
    # Load models
    clf, autoencoder, ood_detector, ae_threshold = load_models()
    if clf is None:
        return
    
    alphabet_files = [f for f in os.listdir(alphabet_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not alphabet_files:
        print(f"\n❌ No images found in '{alphabet_dir}'!")
        return
    
    print(f"\n📝 Testing {len(alphabet_files)} alphabet images:")
    print("-"*70)
    
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
            
            print(f"  {img_file:<25} pred={pred} conf={conf:.2f} | {status}")
        except Exception as e:
            print(f"  {img_file:<25} Error: {e}")
    
    print("-"*70)
    print(f"Summary: {accepted} accepted, {rejected_stage1} rejected at stage 1, {rejected_stage2} rejected at stage 2")
    
    if accepted > 0 and rejected_stage2 == 0:
        print("\n⚠️  No Stage 2 (Mahalanobis) rejections!")
        print("    The thresholds may be too lenient, or alphabet images pass Stage 1 rarely.")

def main():
    """Run all diagnostic tests"""
    print("\n" + "="*70)
    print("CLASS-CONDITIONAL THRESHOLD DIAGNOSTIC TOOL")
    print("="*70)
    
    # 1. Show threshold information
    thresholds = show_threshold_info()
    if thresholds is None:
        return
    
    # 2. Test with noisy versions of valid digits
    test_with_noise()
    
    # 3. Test with alphabet images
    test_alphabet_images()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("1. If you see no Stage 2 rejections:")
    print("   - Thresholds may be too high (very permissive)")
    print("   - Try testing with more challenging OOD samples")
    print("   - Consider using threshold_99 or custom percentile")
    print("\n2. To make thresholds stricter:")
    print("   - Modify nn_train.py to use 90th or 85th percentile")
    print("   - Edit: np.percentile(distances, 90) instead of 95")
    print("\n3. To verify class-conditional behavior:")
    print("   - Look for different thresholds per class (shown above)")
    print("   - Test images that are close to each class boundary")
    print("="*70)

if __name__ == "__main__":
    main()
