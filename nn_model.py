# Shared model architecture for MNIST digit classification
# Import this in both training (torchnn.py) and detection (detect.py)

from torch import nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compact embedding layer (penultimate layer for OOD detection)
        self.embedding = nn.Linear(64*(28-6)*(28-6), 128)
        self.embedding_activation = nn.ReLU()
        
        # Final classification layer
        self.classifier = nn.Linear(128, 10)
    
    def forward(self, x):
        # Full forward pass through all layers
        conv_features = self.conv_layers(x)
        embedding = self.embedding(conv_features)
        embedding = self.embedding_activation(embedding)
        return self.classifier(embedding)
    
    def get_features(self, x):
        """Extract compact 128-d embedding features (penultimate layer)"""
        conv_features = self.conv_layers(x)
        embedding = self.embedding(conv_features)
        embedding = self.embedding_activation(embedding)
        return embedding
