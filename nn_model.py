# Shared model architecture for MNIST digit classification
# Import this in both training (torchnn.py) and detection (detect.py)

from torch import nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor (all layers except final linear)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten()
        )
        # Final classification layer
        self.classifier = nn.Linear(64*(28-6)*(28-6), 10)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def get_features(self, x):
        """Extract features before final classification layer"""
        return self.feature_extractor(x)
