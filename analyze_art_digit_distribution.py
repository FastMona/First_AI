"""Analyze actual digit distribution within ART categories grouped by predicted digit.

For each predicted digit (0-9), shows a histogram of what the actual ground truth
labels were for images assigned to categories mapped to that digit.
"""

import torch
from pathlib import Path
import sys
from collections import defaultdict

from nn_model_art import FuzzyARTMAPClassifier
from config import Config

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.append(str(SRC_DIR))

from first_ai.data import build_mnist_dataloaders  # type: ignore


def analyze_art_digit_distribution():
    """
    Analyze actual digit distribution within categories grouped by predicted digit.
    Shows histograms of ground truth labels for images assigned to each predicted digit.
    """
    
    model_path = Config.MODEL_PATH_ART
    
    if not model_path.exists():
        print(f"❌ ART model not found at {model_path}")
        return
    
    # Load model
    print("Loading ART model...")
    model = FuzzyARTMAPClassifier(
        input_dim=784,
        max_categories=Config.ART_MAX_CATEGORIES,
        vigilance=Config.ART_VIGILANCE,
        learning_rate=Config.ART_LEARNING_RATE,
        choice_alpha=Config.ART_CHOICE_ALPHA,
        count_penalty_gamma=Config.ART_COUNT_PENALTY_GAMMA,
        max_category_count=Config.ART_MAX_CATEGORY_COUNT,
        match_tracking_epsilon=Config.ART_MATCH_TRACKING_EPS,
    )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Load MNIST dataset
    print("Loading MNIST training data...")
    train_loader, val_loader, test_loader = build_mnist_dataloaders(
        dataset_root=Config.DATA_DIR,
        train_batch_size=Config.BATCH_SIZE,
        eval_batch_size=Config.BATCH_SIZE,
        train_ratio=Config.TRAIN_RATIO
    )
    
    # For each predicted digit, track actual labels
    # predicted_digit -> {actual_digit: count}
    digit_distributions = {i: defaultdict(int) for i in range(10)}
    
    print("Analyzing category assignments...")
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).cpu()
            labels = labels.cpu()
            
            # Normalize images
            images = (images - images.min()) / (images.max() - images.min() + 1e-10)
            
            # Get category assignments
            coded_input = model.complement_code(images)
            choice_values = model.category_choice(coded_input, model.committed)
            best_categories = torch.argmax(choice_values, dim=1)
            
            # For each sample, get predicted digit from category and actual digit
            for i in range(images.size(0)):
                cat_idx = best_categories[i].item()
                predicted_digit = model.category_labels[cat_idx].item()
                actual_digit = labels[i].item()
                
                if 0 <= predicted_digit < 10:
                    digit_distributions[predicted_digit][actual_digit] += 1
                    total_samples += 1
    
    # Display results
    print("\n" + "=" * 80)
    print("ART DIGIT DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print("\nFor each predicted digit, showing histogram of actual ground truth labels")
    print(f"Total samples analyzed: {total_samples:,}")
    print("=" * 80)
    
    for predicted_digit in range(10):
        dist = digit_distributions[predicted_digit]
        total_for_digit = sum(dist.values())
        
        if total_for_digit == 0:
            continue
        
        print(f"\n┌─ PREDICTED DIGIT: {predicted_digit} (Total: {total_for_digit:,} samples) " + "─" * 40)
        print("│")
        
        # Calculate accuracy (correct predictions)
        correct = dist[predicted_digit]
        accuracy = (correct / total_for_digit * 100) if total_for_digit > 0 else 0
        
        print(f"│  Accuracy: {accuracy:.1f}% ({correct:,}/{total_for_digit:,} correct)")
        print("│")
        print("│  Actual Label Distribution:")
        
        # Sort by count, descending
        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        
        for actual_digit, count in sorted_dist:
            percentage = (count / total_for_digit * 100) if total_for_digit > 0 else 0
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = "█" * bar_length
            
            # Highlight correct predictions
            marker = "✓" if actual_digit == predicted_digit else " "
            
            print(f"│   {marker} Actual {actual_digit}: {count:6,} ({percentage:5.1f}%) {bar}")
        
        print("└" + "─" * 79)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    total_correct = sum(digit_distributions[d][d] for d in range(10))
    total_samples = sum(sum(dist.values()) for dist in digit_distributions.values())
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    print(f"Correct predictions: {total_correct:,}/{total_samples:,}")
    
    # Per-digit accuracy
    print("\nPer-Digit Accuracy:")
    print(f"{'Digit':<8} {'Accuracy':<12} {'Correct':<12} {'Total':<12}")
    print("-" * 80)
    
    for digit in range(10):
        dist = digit_distributions[digit]
        total_for_digit = sum(dist.values())
        correct = dist[digit]
        accuracy = (correct / total_for_digit * 100) if total_for_digit > 0 else 0
        
        print(f"{digit:<8} {accuracy:>6.1f}%      {correct:>6,}       {total_for_digit:>6,}")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    analyze_art_digit_distribution()
