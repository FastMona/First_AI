"""Analyze Fuzzy ART category distribution across digits."""

import torch
from pathlib import Path
from nn_model_art import FuzzyARTClassifier
from config import Config

def analyze_art_categories():
    """Load trained ART model and display category-to-digit mapping."""
    
    model_path = Config.MODEL_PATH_ART
    
    if not model_path.exists():
        print(f"❌ ART model not found at {model_path}")
        return
    
    # Load model
    model = FuzzyARTClassifier(
        input_dim=784,
        max_categories=Config.ART_MAX_CATEGORIES,
        vigilance=Config.ART_VIGILANCE
    )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Analyze category assignments
    print("\n" + "="*80)
    print("FUZZY ART CATEGORY DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Get category labels and counts
    category_labels = model.category_labels.cpu()
    category_counts = model.category_counts.cpu()
    committed = model.committed.cpu()
    
    # Count categories per digit
    digit_categories = {i: [] for i in range(10)}
    
    for cat_idx in range(model.max_categories):
        if committed[cat_idx]:
            label = category_labels[cat_idx].item()
            if 0 <= label < 10:
                count = category_counts[cat_idx].item()
                digit_categories[label].append({
                    'category_id': cat_idx,
                    'pattern_count': count
                })
    
    # Display results
    total_categories = sum(1 for c in committed if c.item())
    print(f"\nTotal committed categories: {total_categories}/{model.max_categories}")
    print(f"Vigilance parameter: {model.vigilance}")
    
    print("\n" + "-"*80)
    print(f"{'Digit':<8} {'Categories':<15} {'Total Patterns':<15} {'Category Details':<40}")
    print("-"*80)
    
    for digit in range(10):
        cats = digit_categories[digit]
        num_cats = len(cats)
        total_patterns = sum(c['pattern_count'] for c in cats)
        
        # Format category details
        cat_details = ", ".join([f"C{c['category_id']}({c['pattern_count']})" for c in cats])
        
        print(f"{digit:<8} {num_cats:<15} {total_patterns:<15} {cat_details:<40}")
    
    # Summary statistics
    print("-"*80)
    category_counts_list = [len(digit_categories[d]) for d in range(10)]
    total_patterns_all = sum(category_counts[cat_idx].item() for cat_idx in range(model.max_categories) if committed[cat_idx])
    print(f"\n📊 Category Statistics:")
    print(f"  Average categories per digit: {sum(category_counts_list) / 10:.1f}")
    print(f"  Min categories: {min(category_counts_list)}")
    print(f"  Max categories: {max(category_counts_list)}")
    print(f"  Total patterns learned: {total_patterns_all}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    analyze_art_categories()
