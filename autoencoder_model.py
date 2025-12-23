# Autoencoder for MNIST digit reconstruction
# Used as a gate to detect non-digits via reconstruction error

import torch
from torch import nn

class MNISTAutoencoder(nn.Module):
    """Autoencoder trained only on MNIST digits for OOD detection"""
    
    def __init__(self, latent_dim=64):
        super().__init__()
        
        # Encoder: compress 28x28=784 down to latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: reconstruct from latent_dim back to 28x28=784
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()  # Output in [0,1] range like input
        )
    
    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        # Decode
        reconstruction = self.decoder(latent)
        # Reshape back to image
        reconstruction = reconstruction.view(-1, 1, 28, 28)
        return reconstruction
    
    def reconstruction_error(self, x):
        """Compute reconstruction error (MSE) for input"""
        reconstruction = self.forward(x)
        # Mean squared error per image
        error = torch.mean((x - reconstruction)**2, dim=[1,2,3])
        return error
