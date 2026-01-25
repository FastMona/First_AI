"""Camera-based image capture for MNIST model testing.

Captures images from webcam and saves them in the correct format for MNIST model testing.

Dashboard Menu: Called by Option 7 - "Camera Capture"
"""

import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def preprocess_for_mnist(frame, show_steps=False):
    """
    Convert camera frame to MNIST format:
    - Grayscale 28x28 image
    - White digit on black background
    - Normalized [0,1] range
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to get binary image (black/white only)
    # This helps isolate the digit from the background
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to locate the digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assumed to be the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding around the digit
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(thresh.shape[1] - x, w + 2*padding)
        h = min(thresh.shape[0] - y, h + 2*padding)
        
        # Crop to digit region
        digit_roi = thresh[y:y+h, x:x+w]
        
        # Make it square by padding
        max_dim = max(w, h)
        square = np.zeros((max_dim, max_dim), dtype=np.uint8)
        x_offset = (max_dim - w) // 2
        y_offset = (max_dim - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi
    else:
        # If no contour found, use the whole thresholded image
        square = thresh
    
    # Resize to 28x28 (MNIST size)
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Show intermediate steps if requested
    if show_steps:
        cv2.imshow('1. Grayscale', gray)
        cv2.imshow('2. Thresholded', thresh)
        cv2.imshow('3. Final 28x28', cv2.resize(resized, (280, 280), interpolation=cv2.INTER_NEAREST))
    
    return resized

def main():
    # Find the next available image number
    img_counter = 1
    while os.path.exists(f'img_{img_counter}.jpg'):
        img_counter += 1
    
    # Open camera (0 is usually the default camera, try 1 or 2 if 0 doesn't work)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        logger.info("Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            logger.error("Error: No camera found")
            return
    
    # Set camera properties for better responsiveness
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    logger.info("\n" + "="*60)
    logger.info("MNIST Image Capture Tool")
    logger.info("="*60)
    logger.info("Instructions:")
    logger.info("  1. Write a digit (0-9) on WHITE PAPER with a DARK pen")
    logger.info("  2. Hold it up to the camera")
    logger.info("  3. Press SPACE to capture and save as img_X.jpg")
    logger.info("  4. Press 'Q' or 'ESC' to quit")
    logger.info(f"\nNext image will be saved as: img_{img_counter}.jpg")
    logger.info("Preprocessing steps will be shown continuously.")
    logger.info("="*60 + "\n")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Error: Failed to capture frame")
                break
            
            # Display the camera feed
            cv2.imshow('Camera Feed - Press SPACE to capture', frame)
            
            # Always show preprocessing in real-time
            processed = preprocess_for_mnist(frame, show_steps=True)
            
            # Wait for key press (increase to 30ms for better responsiveness)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 is ESC key
                # Quit
                logger.info("\nQuitting...")
                break
            elif key == ord(' '):
                # Space bar: capture and save
                logger.info(f"\nCapturing and saving img_{img_counter}.jpg...")
                
                # Save the processed image
                filename = f'img_{img_counter}.jpg'
                cv2.imwrite(filename, processed)
                logger.info(f"Saved: {filename}")
                
                # Increment counter for next image
                img_counter += 1
                logger.info(f"Next image will be: img_{img_counter}.jpg")
    
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n\nError occurred: {e}")
    finally:
        # Cleanup - ensure this always runs
        logger.info("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        # Force window destruction with a small delay
        cv2.waitKey(1)
        logger.info("Camera closed. Goodbye!")

if __name__ == "__main__":
    main()
