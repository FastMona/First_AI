# Autoencoder for MNIST digit reconstruction
# Used as a gate to detect non-digits via reconstruction error

import torch
from torch import nn
import torch.nn.functional as F

class MNISTAutoencoder(nn.Module):
    """
    Class-conditional autoencoder that learns separate manifolds for each digit.
    
    This implements the biological perception model:
    "I think this is a 3 — does it look like a 3?"
    
    Training: Input (image, label) → learns 10 separate reconstruction manifolds
    Inference: Classifier predicts digit k → reconstruct using manifold k
    """
    
    def __init__(self, latent_dim=64, num_classes=10, embedding_dim=16):
        super().__init__()
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        
        # Label embedding: project class labels into embedding space
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Encoder: compress 28x28=784 + label_embedding down to latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28 + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: reconstruct from latent_dim + label_embedding back to 28x28=784
        # Conditioned on the label to use the appropriate manifold
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()  # Output in [0,1] range like input
        )
    
    def forward(self, x, labels):
        """
        Forward pass with class conditioning.
        
        Args:
            x: Input images [batch_size, 1, 28, 28]
            labels: Class labels [batch_size]
        
        Returns:
            reconstruction: Reconstructed images [batch_size, 1, 28, 28]
        """
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, embedding_dim]
        
        # Flatten image and concatenate with label embedding
        x_flat = x.view(x.size(0), -1)  # [batch_size, 784]
        encoder_input = torch.cat([x_flat, label_emb], dim=1)  # [batch_size, 784 + embedding_dim]
        
        # Encode
        latent = self.encoder(encoder_input)
        
        # Concatenate latent with label embedding for decoding
        decoder_input = torch.cat([latent, label_emb], dim=1)
        
        # Decode
        reconstruction = self.decoder(decoder_input)
        
        # Reshape back to image
        reconstruction = reconstruction.view(-1, 1, 28, 28)
        return reconstruction
    
    def reconstruction_error(self, x, labels):
        """
        Compute class-conditional reconstruction error (MSE) for input.
        
        Args:
            x: Input images [batch_size, 1, 28, 28]
            labels: Predicted class labels to use for reconstruction [batch_size]
        
        Returns:
            error: Reconstruction error per image [batch_size]
        """
        reconstruction = self.forward(x, labels)
        # Mean squared error per image
        error = torch.mean((x - reconstruction)**2, dim=[1,2,3])
        return error
    
    def get_class_specific_error(self, x, predicted_class, return_all=False):
        """
        Compute reconstruction error for a single image using the predicted class manifold.
        Can also compute errors for all class manifolds for comparison.
        
        Args:
            x: Input image [1, 1, 28, 28] or [batch_size, 1, 28, 28]
            predicted_class: The predicted class to use for reconstruction
            return_all: If True, return errors for all classes
        
        Returns:
            error: Reconstruction error using predicted class manifold
            all_errors: (optional) Dict of errors for all classes if return_all=True
        """
        batch_size = x.size(0)
        
        if return_all:
            # Compute reconstruction error for all class manifolds
            all_errors = {}
            for class_idx in range(self.num_classes):
                labels = torch.full((batch_size,), class_idx, dtype=torch.long, device=x.device)
                error = self.reconstruction_error(x, labels)
                all_errors[class_idx] = error.item() if batch_size == 1 else error.cpu().numpy()
            
            # Get the error for the predicted class
            labels = torch.full((batch_size,), predicted_class, dtype=torch.long, device=x.device)
            predicted_error = self.reconstruction_error(x, labels)
            
            return predicted_error, all_errors
        else:
            # Just compute error for predicted class
            labels = torch.full((batch_size,), predicted_class, dtype=torch.long, device=x.device)
            error = self.reconstruction_error(x, labels)
            return error
