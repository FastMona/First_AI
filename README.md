# MNIST Digit Detection with Out-of-Distribution Detection

A PyTorch-based digit classification system with robust out-of-distribution (OOD) detection using a two-stage approach: autoencoder reconstruction and Mahalanobis distance to class prototypes.

## Author

David Cronin 
cronind@sympatico.ca
December 2025

## Project Overview

This project implements a convolutional neural network for MNIST digit classification with advanced OOD detection capabilities. The system can accurately classify handwritten digits (0-9) while rejecting non-digit inputs (letters, symbols, unclear images) through a two-stage validation process.

## Key Features

- ✅ CNN (Convolutional Neural Network) -based digit classifier with early stopping
- ✅ **Class-Conditional Autoencoder** - Biological perception model
- ✅ Two-stage OOD detection:
  - **Stage 1**: Class-conditional reconstruction error (rejects non-reconstructible inputs)
  - **Stage 2**: Diagonal Mahalanobis distance to class prototypes
- ✅ Batch processing capabilities
- ✅ Automated accuracy testing with ground truth validation
- ✅ Visual markdown reports with image thumbnails
- ✅ Manifold visualization and separation analysis

## Class-Conditional Autoencoder (Biological Perception)

### The Concept: "I think this is a 3 — does it look like a 3?"

Traditional autoencoders learn **one global manifold** for all digits, which can make it hard to distinguish between digit classes during OOD detection. Our class-conditional autoencoder implements a more biologically-inspired approach:

**Training Phase:**
- Input: `(image, label)` pairs
- The autoencoder learns **10 separate reconstruction manifolds**
- Each digit learns its own unique geometry
- Example: The "3 manifold" learns what makes a 3 look like a 3

**Inference Phase:**
1. Classifier predicts: "I think this is a 3"
2. Autoencoder reconstructs using the "3 manifold"
3. Check: Does the reconstruction match? → Low error = looks like a 3
4. If reconstruction error is high → doesn't look like a 3 → reject

**Benefits:**
- ✅ Better class separation: Wrong class manifolds produce higher reconstruction error
- ✅ More interpretable: Each class has its own quality check
- ✅ Biological plausibility: Mimics how humans verify perceptions
- ✅ Improved OOD detection: Non-digits fail reconstruction with ALL manifolds

### Manifold Separation

The key advantage is **manifold separation**. When you try to reconstruct:
- A true "3" using the "3 manifold" → **LOW** error
- A true "3" using the "5 manifold" → **HIGH** error
- A random letter using ANY digit manifold → **VERY HIGH** error

This creates natural boundaries between classes and makes OOD detection more robust.

## Python Programs

### Core Training & Models

#### `nn_train.py`
Main training script that:
- Trains the CNN digit classifier on MNIST dataset
- Implements early stopping (monitors validation loss)
- Trains autoencoder for reconstruction-based OOD detection
- Computes class prototypes and Mahalanobis distance parameters
- Saves all trained models (`.pth` files)

**Usage**: `python nn_train.py`

#### `nn_model.py`
CNN model architecture for digit classification:
- 3 convolutional layers with ReLU activation
- Feature extraction and classification layers separated
- Provides `get_features()` method for OOD detection

#### `autoencoder_model.py`
**Class-Conditional Autoencoder** for biological-style perception:
- Learns **10 separate reconstruction manifolds** (one per digit)
- Training: Takes (image, label) pairs → learns digit-specific geometry
- Inference: Uses predicted class to reconstruct → "Does it look like a 3?"
- Architecture:
  - Label embedding: Projects class into 16-dim space
  - Encoder: Compresses 28×28 + label embedding → 64-dim latent
  - Decoder: Reconstructs from latent + label embedding → 28×28
- Trained only on digits to reject non-digit inputs

**Key difference from standard autoencoders:**
- Standard: Learns one global manifold for all digits
- Conditional: Learns 10 separate manifolds, one per digit class
- Better separation: Wrong class manifolds have higher reconstruction error

#### `ood_detector.py`
Mahalanobis distance-based OOD detector:
- Computes distance to class prototypes using diagonal covariance
- Provides calibrated thresholds from training data
- Returns "belongs/doesn't belong" signal for inputs

### Detection Programs

#### `detection_utils.py`
Shared utility functions to eliminate code duplication:
- `load_models()`: Loads all trained models
- `predict_image()`: Two-stage prediction with OOD detection
- `parse_filename()`: Extracts ground truth from test image filenames

#### `detect.py`
Interactive single-image detection:
- Prompts user for image filename
- Displays detailed prediction with confidence scores
- Shows which stage accepted/rejected the input
- Reports reconstruction error and Mahalanobis distance

**Usage**: `python detect.py`

#### `detect_batch.py`
Batch processing for multiple images:
- Processes all images in a specified folder
- Displays summary table with predictions
- Groups results by detected digit
- Lists OOD samples with rejection stage

**Usage**: `python detect_batch.py`

### Testing & Reporting

#### `test_accuracy.py`
Automated accuracy evaluation:
- Tests against labeled images in `test_images/` folder
- Filename convention: `img_X.jpg` where X is digit (0-9) or OOD marker
- Calculates digit classification accuracy and OOD detection accuracy
- Provides overall performance metrics
- Shows breakdown by rejection stage

**Usage**: `python test_accuracy.py`

#### `visualize_conditional_ae.py`
**NEW**: Visualize class-conditional autoencoder manifolds:
- Shows original images with classifier predictions
- Displays reconstructions using ALL 10 class manifolds
- Color-coded: Green = predicted class, Blue = true class
- Bar charts showing reconstruction errors for each manifold
- Manifold separation analysis:
  - Compares correct vs wrong manifold errors
  - Quantifies how well each digit's manifold is separated
  - Higher separation ratio = better class-specific learning

**Demonstrates biological perception**: "I think this is a 3 — does it look like a 3?"

**Usage**: `python visualize_conditional_ae.py`

#### `generate_report.py`
Visual markdown report generator:
- Creates `test_results_report.md` with image thumbnails
- Shows true labels vs predictions in formatted tables
- Separates digit samples from OOD samples
- Includes accuracy statistics and rejection stage breakdown

**Usage**: `python generate_report.py`

### Utilities

#### `clean_project.py`
Project cleanup utility:
- Removes generated files (`.pth` models, `.md` reports)
- Cleans `__pycache__/` folders
- Preserves source code, test images, and MNIST data
- Asks for confirmation before deletion

**Usage**: `python clean_project.py`

## Workflow

### 1. Training
```bash
python nn_train.py
```
This generates:
- `model_state.pth` - Trained classifier weights
- `autoencoder.pth` - Autoencoder with calibrated threshold
- `ood_params.pth` - Class prototypes and covariance parameters

### 2. Testing
```bash
python test_accuracy.py          # Automated accuracy test
python generate_report.py        # Generate visual report
```

### 3. Detection
```bash
python detect.py                 # Single image (interactive)
python detect_batch.py           # Batch processing
```

### 4. Cleanup
```bash
python clean_project.py          # Remove generated files
```

## Directory Structure

```
pytorch_env/
├── nn_train.py                 # Main training script
├── nn_model.py                 # CNN architecture
├── autoencoder_model.py        # Autoencoder for OOD
├── ood_detector.py             # Mahalanobis detector
├── detection_utils.py          # Shared utilities
├── detect.py                   # Single image detection
├── detect_batch.py             # Batch detection
├── test_accuracy.py            # Accuracy testing
├── generate_report.py          # Report generation
├── clean_project.py            # Cleanup utility
├── camera.py                   # [Add description]
├── test_images/                # Test images folder
│   ├── img_0.jpg              # Digit samples (0-9)
│   ├── img_A.jpg              # OOD samples
│   └── ...
├── data/                       # MNIST dataset (auto-downloaded)
│   └── MNIST/
│       └── raw/
└── *.pth                       # Generated model files
```

## Environment Setup

### 1. Activate the conda environment

```powershell
conda activate pytorch
```

### 2. Verify PyTorch + CUDA

```powershell
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 3. Recreate this environment

Reproducible (minimal pinned) environment is in `environment.yml`:

```powershell
conda env create -f environment.yml
conda activate pytorch
```

### 4. VS Code Setup

Set the Python interpreter to:
```
C:\Users\David\Miniconda3\envs\pytorch\python.exe
```

## Requirements

- Python 3.8+
- PyTorch 2.5.1 with CUDA 12.4
- torchvision
- PIL (Pillow)
- numpy

**Hardware**: Compatible with NVIDIA RTX 4070 (driver 591.44)

## OOD Detection Approach

### Two-Stage Detection

**Stage 1: Autoencoder Reconstruction**
- Autoencoder trained only on MNIST digits
- Non-digit inputs produce high reconstruction error
- Threshold calibrated at 95th percentile of training data

**Stage 2: Mahalanobis Distance**
- Measures distance to nearest class prototype
- Uses diagonal covariance (robust for high dimensions)
- Rejects samples far from all digit prototypes

Benefits:
- Stage 1 catches obvious non-digits early
- Stage 2 refines detection for ambiguous cases
- Provides interpretable rejection reasons

## Test Image Naming Convention

Place test images in `test_images/` folder:
- `img_0.jpg` to `img_9.jpg` - Ground truth digit samples
- `img_A.jpg`, `img_+.jpg`, etc. - OOD samples (not digits)

The test programs automatically parse filenames to determine ground truth.

## Performance Metrics

The system reports:
- **Digit Classification Accuracy**: % of digits correctly classified
- **OOD Detection Accuracy**: % of non-digits correctly rejected
- **Overall Accuracy**: Combined performance metric
- **Rejection Stage Breakdown**: How many samples caught at each stage

## Notes

- Models use GPU (CUDA) if available, CPU otherwise
- MNIST dataset (~10MB) downloads automatically on first run
- Training takes ~2-5 minutes on GPU, longer on CPU
- Early stopping prevents overfitting (patience=3 epochs)
- This environment uses PyTorch 2.5.1 built for CUDA 12.4

## Future Improvements

- [ ] Add camera-based real-time detection
- [ ] Implement confidence calibration
- [ ] Add support for other datasets
- [ ] Web interface for detection
- [ ] Ensemble methods for improved accuracy

## License

student code for learning purposes based on public domain information
