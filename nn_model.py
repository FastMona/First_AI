# Shared model architecture for MNIST digit classification
# Import this in both training (torchnn.py) and detection (detect.py)

from torch import nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Using Sequential to create a pipeline where output of one layer feeds into the next
        self.model = nn.Sequential(
            # First Conv2d: 1 input channel (grayscale), 32 output feature maps, 3x3 kernel
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            # Second Conv2d: 32 inputs from previous layer, 64 outputs
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            # Third Conv2d: Keep 64 filters to learn even higher-level features
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            # Flatten converts 2D feature maps into 1D vector for the final classifier
            nn.Flatten(),
            # Final Linear layer: (28-6)*(28-6) comes from 3 conv layers shrinking 28x28 by 2 pixels each
            # 10 outputs = one score for each digit (0-9)
            nn.Linear(64*(28-6)*(28-6), 10)        
        )
    
    def forward(self, x):
        return self.model(x)
