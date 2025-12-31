# CNN model architecture for MNIST digit classification
# Shared between training (nn_train.py) and inference (detect.py, detect_batch.py, etc.)

from torch import nn

class ImageClassifier(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification (0-9).
    
    Architecture:
    - 3 convolutional layers for feature extraction
    - 128-dimensional embedding layer (used for OOD detection)
    - 10-class softmax output
    
    The embedding layer serves dual purposes:
    1. Compact representation for classification
    2. Feature vector for Mahalanobis distance OOD detection
    """
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
        """
        Extract compact 128-d embedding features from penultimate layer.
        
        Used for Mahalanobis distance computation in Stage 2 OOD detection.
        These features capture high-level digit characteristics in a compact space.
        
        Args:
            x: Input images [batch_size, 1, 28, 28]
        
        Returns:
            embedding: Feature vectors [batch_size, 128]
        """
        conv_features = self.conv_layers(x)
        embedding = self.embedding(conv_features)
        embedding = self.embedding_activation(embedding)
        return embedding
