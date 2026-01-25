"""Single-image digit detection program for MNIST model.

Provides interactive interface with detailed two-stage OOD detection output.
Uses shared utilities from detection_utils.py for model loading and predictions.

Dashboard Menu: Called by Option 5 - "Single Image Detection"
"""

import logging
import sys
from detection_utils import (load_models, predict_image, format_detection_result, 
                            print_separator, print_header)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main(image_path=None):
    """Detect digit in a single image
    
    Args:
        image_path (str): Path to image file. If None, will prompt user for input.
    """
    print_header("MNIST Digit Detector with 2-Stage OOD Detection")
    
    # Load models
    logger.info("Loading models...")
    clf, autoencoder, ood_detector, ae_threshold, model_type = load_models()
    
    if clf is None:
        return
    
    logger.info(f"✓ All models loaded successfully! (Using {model_type.upper()} classifier)")
    
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
        
    except ValueError as e:
        # Feature dimension mismatch
        logger.error(f"ERROR: {e}")
    except FileNotFoundError:
        logger.error(f"Image file '{image_path}' not found!")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
