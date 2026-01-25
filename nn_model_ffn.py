"""Feedforward baseline for MNIST classification.

Why: offers a non-convolutional baseline that still emits the same 128-d
embedding for OOD detection, keeping artifacts comparable across model types.
"""

from torch import nn


class FeedforwardClassifier(nn.Module):
    """
    Simple Feedforward Neural Network (Multi-Layer Perceptron) for MNIST classification.
    
    Architecture:
    - Input: Flattened 28x28 image (784 dimensions)
    - Hidden layers: Fully-connected with ReLU activation
    - 128-dimensional embedding layer (for OOD detection consistency)
    - Output: 10-class softmax
    
    This provides a baseline to compare against:
    - CNN: Uses 2D convolutions for spatial feature extraction
    - ART: Uses template matching with resonance search
    - FFN: Simple fully-connected layers with backpropagation (this model)
    """
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256], embedding_size=128, num_classes=10):
        """
        Initialize Feedforward Neural Network.
        
        Args:
            input_size: Flattened input dimension (default 784 for 28x28 MNIST)
            hidden_sizes: List of hidden layer sizes (default [512, 256])
            embedding_size: Embedding layer size (default 128 for OOD detection)
            num_classes: Number of output classes (default 10 for digits 0-9)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        
        # Build hidden layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Dropout for regularization
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Embedding layer (penultimate layer for OOD detection)
        self.embedding = nn.Linear(prev_size, embedding_size)
        self.embedding_activation = nn.ReLU()
        
        # Final classification layer
        self.classifier = nn.Linear(embedding_size, num_classes)
    
    def forward(self, x):
        """
        Full forward pass through all layers.
        
        Args:
            x: Input images [batch_size, 1, 28, 28] or [batch_size, 784]
        
        Returns:
            Class logits [batch_size, 10]
        """
        # Flatten if needed
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        x = self.hidden_layers(x)
        
        # Embedding layer
        embedding = self.embedding(x)
        embedding = self.embedding_activation(embedding)
        
        # Classification
        return self.classifier(embedding)
    
    def get_features(self, x):
        """
        Extract 128-d embedding features from penultimate layer.
        
        Used for Mahalanobis distance computation in Stage 2 OOD detection.
        Maintains consistency with CNN and ART feature extraction.
        
        Args:
            x: Input images [batch_size, 1, 28, 28] or [batch_size, 784]
        
        Returns:
            embedding: Feature vectors [batch_size, 128]
        """
        # Flatten if needed
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        x = self.hidden_layers(x)
        
        # Embedding layer
        embedding = self.embedding(x)
        embedding = self.embedding_activation(embedding)
        
        return embedding
