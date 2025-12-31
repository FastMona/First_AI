# Digit detection program for MNIST model
# Interactive single-image detection with detailed output

from detection_utils import load_models, predict_image
import sys

def main(image_path=None):
    """Detect digit in a single image
    
    Args:
        image_path (str): Path to image file. If None, will prompt user for input.
    """
    print("="*60)
    print("MNIST Digit Detector with 2-Stage OOD Detection")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
    clf, autoencoder, ood_detector, ae_threshold = load_models()
    
    if clf is None:
        return
    
    print("✓ All models loaded successfully!")
    
    # Get image filename from user or parameter
    print("\n" + "-"*60)
    if image_path is None:
        image_path = input("Enter image filename (e.g., test_images/img_1.jpg): ")
    else:
        print(f"Processing image: {image_path}")
    
    try:
        # Make prediction
        prediction, confidence, belongs, recon_error, distance, stage = predict_image(
            image_path, clf, autoencoder, ood_detector, ae_threshold
        )
        
        # Get class-specific threshold
        if ood_detector.class_thresholds_95 and prediction in ood_detector.class_thresholds_95:
            mahal_threshold = ood_detector.class_thresholds_95[prediction]
            threshold_type = f"class-{prediction}"
        else:
            mahal_threshold = ood_detector.threshold_95
            threshold_type = "global"
        
        # Display results
        print("\n" + "="*60)
        
        if not belongs:
            if stage == "reconstruction":
                print("❌ REJECTED AT STAGE 1: RECONSTRUCTION ERROR TOO HIGH")
                print(f"\nThis image cannot be reconstructed as a digit.")
                print(f"Reconstruction error: {recon_error:.6f} (threshold: {ae_threshold:.6f})")
                print(f"\nClassifier's guess: {prediction} ({confidence*100:.1f}%)")
                print("\n💡 Stage 1 Gate: Autoencoder REJECTED this as NOT a digit")
                print("   The autoencoder learned only digits, so it can't recreate this.")
            else:
                print("❌ REJECTED AT STAGE 2: MAHALANOBIS DISTANCE TOO HIGH")
                print(f"\nReconstruction error: {recon_error:.6f} ✓ (passed stage 1)")
                print(f"Mahalanobis distance: {distance:.2f} ✗ ({threshold_type} threshold: {mahal_threshold:.2f})")
                print(f"\nClassifier's guess: {prediction} ({confidence*100:.1f}%)")
                print("\n💡 Stage 1 passed, but Stage 2 Mahalanobis distance REJECTED")
                print("   Image reconstructs OK but doesn't match digit prototypes.")
        else:
            print(f"✓ PASSED BOTH STAGES - VALID DIGIT")
            print(f"\n🔢 Predicted Digit: {prediction}")
            print(f"   Confidence: {confidence*100:.1f}%")
            print(f"\nStage 1 - Reconstruction error: {recon_error:.6f} ✓ (threshold: {ae_threshold:.6f})")
            print(f"Stage 2 - Mahalanobis distance: {distance:.2f} ✓ ({threshold_type} threshold: {mahal_threshold:.2f})")
            
            relative_recon = recon_error / ae_threshold * 100
            relative_mahal = distance / mahal_threshold * 100
            
            if relative_recon < 50 and relative_mahal < 50:
                print(f"\n💪 Excellent digit - very typical example!")
            elif relative_recon < 75 and relative_mahal < 75:
                print(f"\n✓ Good digit - normal example")
            else:
                print(f"\n⚠️ Acceptable but somewhat atypical")
        
        print("="*60)
        
    except FileNotFoundError:
        print(f"\nError: Image file '{image_path}' not found!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
