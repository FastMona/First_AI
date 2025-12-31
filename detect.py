# Single-image digit detection program for MNIST model
# Interactive interface with detailed two-stage OOD detection output
# Uses shared utilities from detection_utils.py

from detection_utils import (load_models, predict_image, format_detection_result, 
                            print_separator, print_header)
import sys

def main(image_path=None):
    """Detect digit in a single image
    
    Args:
        image_path (str): Path to image file. If None, will prompt user for input.
    """
    print_header("MNIST Digit Detector with 2-Stage OOD Detection")
    
    # Load models
    print("\nLoading models...")
    clf, autoencoder, ood_detector, ae_threshold = load_models()
    
    if clf is None:
        return
    
    print("✓ All models loaded successfully!")
    
    # Get image filename from user or parameter
    print_separator('-')
    if image_path is None:
        image_path = input("Enter image filename (e.g., test_images/img_1.jpg): ")
    else:
        print(f"Processing image: {image_path}")
    
    try:
        # Make prediction
        prediction, confidence, belongs, recon_error, distance, stage = predict_image(
            image_path, clf, autoencoder, ood_detector, ae_threshold
        )
        
        # Display formatted results using utility function
        result_message = format_detection_result(
            prediction, confidence, belongs, recon_error, distance,
            stage, ood_detector, ae_threshold, verbose=True
        )
        print("\n" + result_message)
        
    except FileNotFoundError:
        print(f"\nError: Image file '{image_path}' not found!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
