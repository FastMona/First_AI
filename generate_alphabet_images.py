"""
Generate 28x28 pixel images of alphabet letters (A-Z) in .jpg format
"""
import os
from PIL import Image, ImageDraw, ImageFont
import string

# Create output directory
output_dir = "alphabet_images"
os.makedirs(output_dir, exist_ok=True)

# Image settings
img_size = (28, 28)
background_color = 0  # Black (0 for grayscale)
text_color = 255  # White (255 for grayscale)

# Try to use a default font, fallback if not available
try:
    # Try to use a TrueType font with appropriate size
    font = ImageFont.truetype("arial.ttf", 20)
except:
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        # Use default font if TrueType fonts not available
        font = ImageFont.load_default()
        print("Using default font - letters may appear smaller")

# Generate images for A-Z (uppercase)
for letter in string.ascii_uppercase:
    # Create blank image in grayscale mode (L) to match MNIST format
    img = Image.new('L', img_size, background_color)
    draw = ImageDraw.Draw(img)
    
    # Get text bounding box to center the letter
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position to center the text
    x = (img_size[0] - text_width) // 2 - bbox[0]
    y = (img_size[1] - text_height) // 2 - bbox[1]
    
    # Draw the letter
    draw.text((x, y), letter, fill=text_color, font=font)
    
    # Save as .jpg
    filename = f"img_{letter}.jpg"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath, "JPEG")
    print(f"Created: {filename}")

print(f"\nSuccessfully created {len(string.ascii_uppercase)} alphabet images in '{output_dir}/' folder")
print("Image format: 28x28 pixels, .jpg format")
