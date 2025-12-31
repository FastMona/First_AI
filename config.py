# Configuration file for MNIST digit detection project
# Centralizes all magic numbers, paths, and hyperparameters

import torch

class Config:
    """
    Global configuration for MNIST digit detection project.
    
    Centralizes all hyperparameters, paths, and constants to ensure:
    - Single source of truth (no magic numbers)
    - Easy tuning and experimentation
    - Consistent values across all modules
    """
    
    # Device configuration - automatically detects CUDA GPU if available
    # All models and tensors should use Config.DEVICE instead of hardcoding 'cuda'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    CNN_EPOCHS = 10
    AE_EPOCHS = 5
    EARLY_STOP_PATIENCE = 3
    
    # Dataset split ratios
    TRAIN_RATIO = 0.8  # 80% train, 20% validation
    
    # Model architecture
    FEATURE_DIM = 128  # Compact embedding dimension for CNN
    LATENT_DIM = 64    # Autoencoder latent dimension
    EMBEDDING_DIM = 16 # Label embedding dimension for autoencoder
    NUM_CLASSES = 10   # Number of digit classes (0-9)
    
    # CNN architecture
    CONV_CHANNELS = [32, 64, 64]
    CONV_KERNEL = (3, 3)
    INPUT_SIZE = 28  # MNIST image size
    
    # Autoencoder architecture
    AE_HIDDEN_LAYERS = [256, 128]  # Encoder/decoder hidden layers
    
    # Model file paths
    MODEL_PATH = 'model_state.pth'
    AUTOENCODER_PATH = 'autoencoder.pth'
    OOD_PARAMS_PATH = 'ood_params.pth'
    
    # Data paths
    DATA_DIR = 'data'
    TEST_IMAGES_DIR = 'test_images'
    
    # OOD detection thresholds (percentiles of in-distribution distances)
    # Lower percentile = stricter rejection = fewer false acceptances (recommended: 90)
    # Higher percentile = lenient acceptance = fewer false rejections
    DEFAULT_PERCENTILE = 90  # Stricter - rejects more borderline cases
    PERCENTILE_90 = 90       # Recommended for production
    PERCENTILE_95 = 95       # Moderate balance
    PERCENTILE_99 = 99       # Lenient - accepts more edge cases
    
    # Regularization
    COVARIANCE_REG = 1e-4  # Small epsilon to prevent division by zero in covariance
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Image processing
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # Display formatting
    SEPARATOR_WIDTH = 80
    SEPARATOR_CHAR = '='
    SUB_SEPARATOR_CHAR = '-'
    
    @classmethod
    def get_device_info(cls):
        """Return human-readable device information (e.g., 'CUDA (NVIDIA RTX 3080)' or 'CPU')"""
        if cls.DEVICE.type == 'cuda':
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        return "CPU"
