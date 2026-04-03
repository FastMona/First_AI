"""Neocognitron-inspired classifier for MNIST.

Why: adds a biologically inspired S/C stage architecture (Fukushima) while
preserving the same 128-d feature API used by the OOD pipeline.
"""

from torch import nn


class NeocognitronClassifier(nn.Module):
    """
    Neocognitron-inspired neural network for MNIST classification.

    The original Neocognitron (Kunihiko Fukushima) alternates between:
    - S-cells: feature-selective simple units
    - C-cells: local pooling/position-tolerant complex units

    This implementation mirrors that structure with modern PyTorch modules:
    - S stages: Conv2d + ReLU + LocalResponseNorm (competition-like behavior)
    - C stages: AvgPool2d (local shift tolerance)

    Architecture (28x28 input):
    - S1/C1: 1 -> 32 channels
    - S2/C2: 32 -> 64 channels
    - S3/C3: 64 -> 96 channels + global pooling
    - 128-d embedding (for OOD Mahalanobis compatibility)
    - 10-class classifier head
    """

    def __init__(self, num_classes=10, embedding_dim=128):
        super().__init__()

        # S1/C1 stage
        self.s1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
        )
        self.c1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # S2/C2 stage
        self.s2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
        )
        self.c2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # S3/C3 stage
        self.s3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
        )
        self.c3 = nn.AdaptiveAvgPool2d((1, 1))

        # Penultimate embedding and final classifier
        self.embedding = nn.Linear(96, embedding_dim)
        self.embedding_activation = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _extract_backbone(self, x):
        """Run Neocognitron-style S/C stages and return flattened backbone features."""
        x = self.s1(x)
        x = self.c1(x)

        x = self.s2(x)
        x = self.c2(x)

        x = self.s3(x)
        x = self.c3(x)

        return x.view(x.size(0), -1)

    def get_features(self, x):
        """
        Extract 128-d embedding features from the penultimate layer.

        Args:
            x: Input images [batch_size, 1, 28, 28]

        Returns:
            embedding: Feature vectors [batch_size, embedding_dim]
        """
        backbone_features = self._extract_backbone(x)
        embedding = self.embedding(backbone_features)
        embedding = self.embedding_activation(embedding)
        return embedding

    def forward(self, x):
        """
        Full forward pass through Neocognitron-inspired backbone and classifier.

        Args:
            x: Input images [batch_size, 1, 28, 28]

        Returns:
            logits: Class scores [batch_size, num_classes]
        """
        embedding = self.get_features(x)
        return self.classifier(embedding)
