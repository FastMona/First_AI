"""Camera-based image capture for MNIST model testing.

Captures images from webcam and saves them in the correct format for MNIST model testing.

Dashboard Menu: Called by Option 7 - "Camera Capture"
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path

from config import Config

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
    # Ensure captures directory exists
    capture_dir = Config.CAPTURES_DIR
    capture_dir.mkdir(parents=True, exist_ok=True)

    # Find the next available image number in captures/
    existing = sorted(capture_dir.glob("capture_*.jpg"))
    if existing:
        last = existing[-1].stem.split("_")[-1]
        try:
            img_counter = int(last) + 1
        except ValueError:
            img_counter = len(existing) + 1
    else:
        img_counter = 1
    
    # Open camera (0 is usually the default camera, try 1 or 2 if 0 doesn't work)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: No camera found")
            return
    
    # Set camera properties for better responsiveness
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("\n" + "="*60)
    print("MNIST Image Capture Tool")
    print("="*60)
    print("Instructions:")
    print("  1. Write a digit (0-9) on WHITE PAPER with a DARK pen")
    print("  2. Hold it up to the camera")
    print("  3. Press SPACE to capture and save as img_X.jpg")
    print("  4. Press 'Q' or 'ESC' to quit")
    print(f"\nNext image will be saved as: {capture_dir / f'capture_{img_counter:04d}.jpg'}")
    print("Preprocessing steps will be shown continuously.")
    print("="*60 + "\n")
    print("IMPORTANT: Click on one of the OpenCV windows to give it focus!")
    print("If keyboard input doesn't work, try clicking the window again.\n")
    
    # Create all windows upfront
    cv2.namedWindow('Camera Feed - Press SPACE to capture', cv2.WINDOW_NORMAL)
    cv2.namedWindow('1. Grayscale', cv2.WINDOW_NORMAL)
    cv2.namedWindow('2. Thresholded', cv2.WINDOW_NORMAL)
    cv2.namedWindow('3. Final 28x28', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Display the camera feed
            cv2.imshow('Camera Feed - Press SPACE to capture', frame)
            
            # Always show preprocessing in real-time
            processed = preprocess_for_mnist(frame, show_steps=True)
            
            # Wait for key press with longer delay and force window update
            # Use 100ms for better keyboard responsiveness on Windows
            key = cv2.waitKey(100) & 0xFF
            
            # Skip if no key was pressed
            if key == 255:
                continue
            
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 is ESC key
                # Quit
                print("\nQuitting...")
                break
            elif key == ord(' '):
                # Space bar: capture and save
                print(f"\nCapturing and saving capture_{img_counter:04d}.jpg...")
                
                # Save the processed image
                filename = capture_dir / f"capture_{img_counter:04d}.jpg"
                cv2.imwrite(str(filename), processed)
                print(f"Saved: {filename}")
                
                # Increment counter for next image
                img_counter += 1
                print(f"Next image will be: {capture_dir / f'capture_{img_counter:04d}.jpg'}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
    finally:
        # Cleanup - ensure this always runs
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        # Force window destruction with a small delay
        cv2.waitKey(1)
        print("Camera closed. Goodbye!")

if __name__ == "__main__":
    main()
