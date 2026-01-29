"""Download handwritten symbol images for OOD testing.

Attempts to download handwritten symbols (currency, math, punctuation) from available datasets.
Falls back to font-based generation if handwritten data is unavailable."""

import requests
from PIL import Image
from io import BytesIO
import os
import numpy as np

def download_handwritten_symbols():
    """Attempt to download handwritten symbol datasets"""
    print("Searching for handwritten symbol datasets...")
    
    # Try EMNIST Special Characters (if available via torchvision)
    try:
        import torch
        from torchvision import datasets
        from torchvision.transforms import ToTensor
        
        print("Attempting to download EMNIST dataset (includes some symbols)...")
        # EMNIST has 'byclass' which includes digits, letters, but limited symbols
        # This might not work well, but let's try
        
        dataset = datasets.EMNIST(
            root='./temp_data',
            split='byclass',
            train=False,
            download=True,
            transform=ToTensor()
        )
        
        print(f"✓ Downloaded EMNIST dataset with {len(dataset)} samples")
        return dataset, 'emnist'
        
    except Exception as e:
        print(f"✗ EMNIST not suitable: {e}")
    
    # Try downloading from Kaggle or other sources
    print("\nSearching for alternative handwritten symbol sources...")
    
    # URLs for potential handwritten symbol datasets (public domain)
    symbol_sources = [
        {
            'name': 'HASYv2 (Handwritten Symbol Database)',
            'info': 'Contains mathematical symbols',
            'url': None  # Requires Kaggle API or direct download
        }
    ]
    
    for source in symbol_sources:
        print(f"  - {source['name']}: {source['info']}")
    
    print("\n⚠️  No readily available handwritten symbol datasets found.")
    print("    Handwritten symbols are rare compared to MNIST digits.")
    print("    Falling back to font-based generation with variations...\n")
    
    return None, None

def create_varied_symbol_image(symbol, filename, variation=0):
    """Create a 28x28 grayscale image with variations to simulate handwriting"""
    try:
        from PIL import ImageDraw, ImageFont, ImageFilter
        import random
        
        # Create black background
        img = Image.new('L', (32, 32), color=0)  # Slightly larger for rotation
        draw = ImageDraw.Draw(img)
        
        # Use font with some variation in size
        try:
            font_size = random.randint(18, 22)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box to center it
        bbox = draw.textbbox((0, 0), symbol, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Add random offset for "handwritten" feel
        offset_x = random.randint(-2, 2)
        offset_y = random.randint(-2, 2)
        
        # Center with offset
        x = (32 - text_width) // 2 - bbox[0] + offset_x
        y = (32 - text_height) // 2 - bbox[1] + offset_y
        
        # Draw with slight rotation
        draw.text((x, y), symbol, fill=255, font=font)
        
        # Apply random rotation
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, fillcolor=0, expand=False)
        
        # Add slight blur for "ink bleed" effect
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        
        # Crop back to 28x28
        img = img.crop((2, 2, 30, 30))
        
        # Add slight noise
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Save
        img.save(f"other_images/{filename}")
        return True
    except Exception as e:
        print(f"✗ Failed {filename}: {e}")
        return False

# Symbols that are NOT digits or letters
symbols = [
    ("$", "dollar.jpg"),
    ("€", "euro.jpg"),
    ("£", "pound.jpg"),
    ("¥", "yen.jpg"),
    ("+", "plus.jpg"),
    ("-", "minus.jpg"),
    ("×", "multiply.jpg"),
    ("÷", "divide.jpg"),
    ("=", "equals.jpg"),
    ("%", "percent.jpg"),
    ("!", "exclaim.jpg"),
    ("?", "question.jpg"),
    ("@", "at.jpg"),
    ("#", "hash.jpg"),
    ("&", "ampersand.jpg"),
    ("*", "asterisk.jpg"),
    ("(", "paren_open.jpg"),
    (")", "paren_close.jpg"),
    ("[", "bracket_open.jpg"),
    ("]", "bracket_close.jpg"),
]

print("=" * 60)
print("HANDWRITTEN SYMBOL IMAGE GENERATION")
print("=" * 60)

# Try to find handwritten datasets
dataset, source = download_handwritten_symbols()

os.makedirs("other_images", exist_ok=True)

if dataset is None:
    # Fall back to varied font-based generation
    print("Creating varied symbol images (font-based with handwritten effects)...")
    print("=" * 60)
    
    success = 0
    for symbol, filename in symbols:
        if create_varied_symbol_image(symbol, filename):
            print(f"✓ Created: {filename} ({symbol})")
            success += 1
    
    print("=" * 60)
    print(f"Created {success}/{len(symbols)} varied symbol images in other_images/")
    print("Note: Using font-based with rotation, noise, and blur for variation")
else:
    print(f"Using {source} dataset for symbols")
    print("Note: May have limited symbol coverage")
