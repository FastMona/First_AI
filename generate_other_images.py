# Generate 28x28 images of Unicode symbols (not letters/numbers)
# White symbols on black background for OOD testing

from PIL import Image, ImageDraw, ImageFont
import os

def create_symbol_image(symbol, filename, size=28):
    """Create a 28x28 image with a white symbol on black background"""
    # Create black background
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)
    
    # Try to use a font that supports Unicode
    try:
        # Try different font sizes to fit the symbol
        for font_size in [20, 18, 16, 14, 12]:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
                break
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    break
                except:
                    font = ImageFont.load_default()
                    break
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box to center it
    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (size - text_width) // 2 - bbox[0]
    y = (size - text_height) // 2 - bbox[1]
    
    # Draw white text on black background
    draw.text((x, y), symbol, fill=255, font=font)
    
    # Save image
    img.save(filename)
    print(f"Created: {filename} ({symbol})")

def main():
    # Create output directory
    output_dir = 'other_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Unicode symbols (not letters or numbers)
    symbols = [
        ('!', 'exclamation'),
        ('?', 'question'),
        ('@', 'at'),
        ('#', 'hash'),
        ('$', 'dollar'),
        ('%', 'percent'),
        ('&', 'ampersand'),
        ('*', 'asterisk'),
        ('+', 'plus'),
        ('-', 'minus'),
        ('=', 'equals'),
        ('/', 'slash'),
        ('\\', 'backslash'),
        ('|', 'pipe'),
        ('~', 'tilde'),
        ('^', 'caret'),
        ('_', 'underscore'),
        ('(', 'lparen'),
        (')', 'rparen'),
        ('{', 'lbrace'),
        ('}', 'rbrace'),
        ('[', 'lbracket'),
        (']', 'rbracket'),
        ('<', 'less'),
        ('>', 'greater'),
    ]
    
    print(f"Generating {len(symbols)} symbol images in '{output_dir}/'...")
    print("="*60)
    
    for symbol, name in symbols:
        filename = os.path.join(output_dir, f'img_{name}.jpg')
        try:
            create_symbol_image(symbol, filename)
        except Exception as e:
            print(f"Error creating {name}: {e}")
    
    print("="*60)
    print(f"✓ Generated {len(symbols)} symbol images")
    print(f"  Location: {output_dir}/")
    print(f"  Format: 28x28 grayscale, white on black")

if __name__ == "__main__":
    main()
