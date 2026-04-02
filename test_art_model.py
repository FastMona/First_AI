"""Quick test of ART model state"""
import torch
from nn_model_art import FuzzyARTMAPClassifier
from config import Config

model = FuzzyARTMAPClassifier(
    784,
    Config.ART_MAX_CATEGORIES,
    Config.ART_VIGILANCE,
    Config.ART_LEARNING_RATE,
    Config.ART_CHOICE_ALPHA,
    Config.ART_COUNT_PENALTY_GAMMA,
    Config.ART_MAX_CATEGORY_COUNT,
    Config.ART_MATCH_TRACKING_EPS,
)

checkpoint = torch.load(Config.MODEL_PATH_ART, weights_only=False, map_location='cpu')
model.load_state_dict(checkpoint)

print(f"Model loaded successfully")
print(f"Committed categories: {model.num_committed}")
print(f"Total categories: {model.committed.sum().item()}")
print(f"\nFirst 10 category labels: {model.category_labels[:10].tolist()}")
print(f"First 10 category counts: {model.category_counts[:10].tolist()}")

# Test with a random input
test_input = torch.rand(1, 784)
logits = model.predict(test_input)
print(f"\nTest prediction logits: {logits}")
print(f"Predicted class: {logits.argmax(dim=1).item()}")
