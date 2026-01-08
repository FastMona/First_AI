# MNIST Digit Detection with Out-of-Distribution Detection

A PyTorch-based digit classification system with robust out-of-distribution (OOD) detection using a two-stage approach: autoencoder reconstruction and Mahalanobis distance to class prototypes.

## Quick Start

**Run the Dashboard (Recommended):**

```bash
python dashboard.py
```

The dashboard provides easy access to all project functionalities through an interactive menu.

## Author

David Cronin
<cronind@sympatico.ca>
December 2025

## Project Overview

This project implements a convolutional neural network for MNIST digit classification with advanced out-of-distribution (OOD) detection capabilities. The system accurately classifies handwritten digits (0-9) while rejecting non-digit inputs (letters, symbols, corrupted images) through a **two-stage validation process**:

1. **Stage 1 - Reconstruction Gate**: Class-conditional autoencoder checks if input can be reconstructed as the predicted digit
2. **Stage 2 - Prototype Distance**: Mahalanobis distance verifies input is close to digit class prototypes

The project features a **maintainable, modular codebase** with centralized configuration and shared utilities.

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

- Trains the CNN digit classifier on MNIST dataset (10 epochs max)
- Implements early stopping with patience=3 (monitors test loss)
- Trains class-conditional autoencoder for Stage 1 OOD detection (5 epochs)
- Computes class prototypes and Mahalanobis distance parameters from **validation set**
- Calibrates OOD thresholds at 90th/95th/99th percentiles (validation data)
- Saves all trained models and parameters (`.pth` files)

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

Mahalanobis distance-based OOD detector (Stage 2):

- Computes distance to class prototypes using diagonal covariance matrix
- Supports class-conditional thresholds (90th/95th/99th percentiles)
- Hierarchical threshold selection: Class-90th > Class-95th > Global-95th
- Returns "belongs/doesn't belong" signal with distance metrics

#### `config.py`

Centralized configuration module:

- All hyperparameters (batch size, learning rates, epochs, thresholds)
- Model architecture constants (feature dimensions, layer sizes)
- File paths and data directories
- Auto-detects CUDA vs CPU device
- Single source of truth - eliminates magic numbers

### Detection Programs

#### `detection_utils.py`

Shared utility functions to eliminate code duplication:

- `load_models()`: Loads classifier, autoencoder, and OOD detector from config paths
- `predict_image()`: Two-stage prediction pipeline with OOD detection
- `get_class_threshold()`: Hierarchical threshold selection logic
- `format_detection_result()`: Unified result formatting (verbose/compact modes)
- `parse_filename()`: Extracts ground truth labels from test filenames
- `print_header()`, `print_separator()`: Consistent display formatting

Used by: `detect.py`, `detect_batch.py`, `test_accuracy.py`, `generate_report.py`

#### `detect.py`

Interactive single-image detection:

- Prompts user for image filename (or pass as argument)
- Uses `format_detection_result()` for consistent, detailed output
- Shows classifier prediction with confidence
- Reports Stage 1 reconstruction error vs threshold
- Reports Stage 2 Mahalanobis distance vs class-conditional threshold
- Indicates rejection stage and reasoning

**Usage**:

```bash
python detect.py                    # Interactive mode
python detect.py test_images/img_3.jpg  # Direct file
```

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

Visualize class-conditional autoencoder manifolds:

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

#### `dashboard.py`

Interactive menu system for all project functionality:

- Easy navigation to all features
- Environment validation (checks PyTorch, CUDA)
- Training, detection, testing, and visualization options
- Integrated cleanup utility

**Usage**: `python dashboard.py` (Recommended entry point)

#### `camera.py`

Webcam-based image capture utility:

- Captures images from default camera
- Preprocesses to MNIST format (28×28 grayscale)
- Applies adaptive thresholding for digit isolation
- Saves as `img_X.jpg` for testing

**Usage**: `python camera.py`

#### `clean_project.py`

Project cleanup utility:

- Removes generated files (`.pth` models, `.md` reports, `.png` plots)
- Cleans `__pycache__/` folders
- Preserves source code, test images, and MNIST data
- Interactive or automatic mode

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

```text
pytorch_env/
├── config.py                   # Centralized configuration
├── nn_train.py                 # Main training script
├── nn_model.py                 # CNN architecture
├── autoencoder_model.py        # Class-conditional autoencoder
├── ood_detector.py             # Mahalanobis OOD detector (Stage 2)
├── detection_utils.py          # Shared utilities
├── detect.py                   # Single image detection
├── detect_batch.py             # Batch detection
├── test_accuracy.py            # Accuracy testing
├── generate_report.py          # Report generation
├── visualize_conditional_ae.py # Manifold visualization
├── dashboard.py                # Interactive menu system
├── clean_project.py            # Cleanup utility
├── camera.py                   # Webcam capture for testing
├── test_images/                # Test images folder
│   ├── img_0.jpg              # Digit samples (0-9)
│   ├── img_A.jpg              # OOD samples
│   └── ...
├── training_data/              # MNIST dataset (auto-downloaded)
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

```text
C:\Users\David\Miniconda3\envs\pytorch\python.exe
```

## Model Architecture Details

### CNN Classifier

- **Input**: 28×28 grayscale images
- **Convolutional layers**: 3 layers (32→64→64 channels, 3×3 kernels)
- **Embedding layer**: 128-dimensional feature vector (penultimate layer)
- **Output**: 10-class softmax (digits 0-9)
- **Total parameters**: ~150K

### Class-Conditional Autoencoder

- **Input**: 28×28 image + class label (0-9)
- **Label embedding**: 16-dimensional learned embedding per class
- **Encoder**: 784+16 → 256 → 128 → 64 (latent)
- **Decoder**: 64+16 → 128 → 256 → 784
- **Output**: Reconstructed 28×28 image
- **Training**: 10 separate manifolds, one per digit class

### Data Split

- **Training**: 48,000 samples (80% of MNIST train) - used for model training
- **Validation**: 12,000 samples (20% of MNIST train) - used for threshold calibration
- **Test**: 10,000 samples (MNIST test set) - used for final evaluation

## Requirements

- Python 3.8+
- PyTorch 2.5.1 with CUDA 12.4
- torchvision
- PIL (Pillow)
- numpy
- matplotlib (for visualizations)

**Hardware**: Compatible with NVIDIA RTX 4070 (driver 591.44)
**Device**: Auto-detects CUDA GPU, falls back to CPU

## OOD Detection Approach

### Two-Stage Detection

#### Stage 1: Autoencoder Reconstruction Gate

- Class-conditional autoencoder trained only on MNIST digits (10 manifolds)
- Non-digit inputs produce high reconstruction error
- Threshold calibrated at **95th percentile of validation data** (using predicted labels)
- Fast rejection: catches obvious non-digits early

#### Stage 2: Mahalanobis Distance to Prototypes

- Measures distance from 128-d feature embedding to nearest class prototype
- Uses **diagonal covariance matrix** (faster, robust for high dimensions)
- **Class-conditional thresholds** at 90th percentile (stricter, default)
- Hierarchical fallback: 90th > 95th > global threshold
- Rejects samples far from all digit prototypes

Benefits:

- Stage 1 catches obvious non-digits early
- Stage 2 refines detection for ambiguous cases
- Provides interpretable rejection reasons

## Test Image Naming Convention

Place test images in `test_images/` folder with this naming scheme:

**Digit Samples** (for classification accuracy):

- `img_0.jpg` to `img_9.jpg` - Images containing digits 0-9
- Example: `img_3.jpg` should contain the digit "3"

**OOD Samples** (for rejection accuracy):

- `img_A.jpg`, `img_letter.jpg` - Letters
- `img_+.jpg`, `img_symbol.png` - Symbols  
- `img_cat.jpg`, `img_noise.png` - Non-digit images

The filename pattern `img_X.ext` allows automated testing:

- Single digit X (0-9) → Ground truth digit label
- Non-digit X → OOD sample (should be rejected)

Test programs automatically parse filenames to compute accuracy metrics.

## Performance Metrics

The system reports comprehensive accuracy metrics:

### Classification Performance

- **Digit Classification Accuracy**: Percentage of true digits (0-9) correctly classified
- **Confidence Scores**: Softmax probability for predicted class
- **Per-Class Accuracy**: Breakdown by individual digits

### OOD Detection Performance

- **OOD Detection Accuracy**: Percentage of non-digits correctly rejected
- **False Acceptance Rate**: Non-digits incorrectly classified as digits
- **False Rejection Rate**: True digits incorrectly rejected as OOD
- **Overall Accuracy**: Combined metric (correct classifications + correct rejections)

### Stage Analysis

- **Stage 1 Rejections**: Count rejected by autoencoder (reconstruction error)
- **Stage 2 Rejections**: Count rejected by Mahalanobis distance
- **Stage Distribution**: Where in the pipeline samples are caught

### Threshold Calibration

- **90th Percentile** (default): Stricter, fewer false acceptances, more false rejections
- **95th Percentile**: Balanced tradeoff
- **99th Percentile**: Lenient, fewer false rejections, more false acceptances

Typical performance:

- Digit accuracy: 98-99% (MNIST digits)
- OOD accuracy: 95-98% (non-digit rejection)
- Overall: 97-99% depending on threshold choice

## Notes

- **Device Selection**: Models automatically use CUDA GPU if available, otherwise CPU
- **MNIST Dataset**: ~10MB, downloads automatically on first run to `training_data/` folder
- **Training Time**:
  - GPU (RTX 4070): ~2-3 minutes total
  - CPU: ~15-20 minutes total
- **Early Stopping**: Prevents overfitting with patience=3 epochs (monitors test loss)
- **Threshold Calibration**: Uses validation set (not training) for realistic thresholds
- **Diagonal Covariance**: Faster than full covariance with acceptable accuracy loss
- **Class-Conditional Thresholds**: Each digit (0-9) has its own threshold for better precision
- **PyTorch Version**: 2.5.1 built for CUDA 12.4 (compatible with driver 591.44+)

## Recent Refactoring (December 2025)

### Code Quality Improvements

✅ **Centralized Configuration** - Created `config.py` to eliminate magic numbers
✅ **Fixed Critical Bugs** - Removed syntax errors and duplicate code in `ood_detector.py`, `nn_train.py`
✅ **Extracted Common Logic** - Added utility functions to `detection_utils.py`:

- `get_class_threshold()` - Hierarchical threshold selection
- `format_detection_result()` - Unified output formatting
- Display helpers for consistent UI

✅ **Enhanced Documentation** - Comprehensive docstrings and comments explaining:

- Algorithm choices (diagonal covariance, percentile thresholds)
- Biological perception model
- Two-stage detection rationale

✅ **Improved Maintainability** - Single source of truth, DRY principle, clear separation of concerns

### Architecture Benefits

- **Modularity**: Clear separation between models, detection logic, and utilities
- **Consistency**: All scripts use shared config and utilities
- **Testability**: Isolated functions easier to test and debug
- **Extensibility**: Easy to add new features without duplicating code

## Future Improvements

- [ ] Real-time detection pipeline
- [ ] Confidence calibration (Platt scaling)
- [ ] Support for other datasets (Fashion-MNIST, custom digits)
- [ ] Web interface with REST API
- [ ] Ensemble methods (multiple models)
- [ ] Type hints for all functions
- [ ] Unit tests with pytest
- [ ] Docker containerization

## License

student code for learning purposes based on public domain information
