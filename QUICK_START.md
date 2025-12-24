# Class-Conditional Autoencoder - Quick Start Guide

## What Was Implemented

Your MNIST digit detection system now uses a **class-conditional autoencoder** that implements biological perception: "I think this is a 3 — does it look like a 3?"

### Key Changes

1. **Autoencoder learns 10 separate manifolds** (one per digit 0-9)
2. **Training**: Input is (image, label) → learns digit-specific reconstruction
3. **Inference**: Uses classifier prediction to select which manifold to use
4. **Better OOD detection**: Wrong class manifolds produce higher errors

## Files Modified

### Core Implementation
- ✅ [autoencoder_model.py](autoencoder_model.py) - Class-conditional architecture with label embedding
- ✅ [nn_train.py](nn_train.py) - Training updated to pass labels
- ✅ [detection_utils.py](detection_utils.py) - Inference uses predicted class manifold

### New Files
- ✅ [visualize_conditional_ae.py](visualize_conditional_ae.py) - Visualize manifold separation
- ✅ [test_conditional_ae_architecture.py](test_conditional_ae_architecture.py) - Architecture validation
- ✅ [CLASS_CONDITIONAL_IMPLEMENTATION.md](CLASS_CONDITIONAL_IMPLEMENTATION.md) - Detailed documentation
- ✅ [QUICK_START.md](QUICK_START.md) - This file

### Documentation
- ✅ [README.md](README.md) - Updated with class-conditional explanation

## Quick Start

### Step 1: Verify Architecture ✓ (Already Done)
```bash
conda activate pytorch
python test_conditional_ae_architecture.py
```
**Status**: ✅ PASSED - Architecture is working correctly!

### Step 2: Train Models
```bash
python nn_train.py
```
This will:
- Train CNN classifier
- Train class-conditional autoencoder (5 epochs with label conditioning)
- Calibrate thresholds using predicted labels
- Save models to `.pth` files

Expected output:
```
Training Class-Conditional Autoencoder
Learning 10 separate digit manifolds (one per class)
Biological perception: 'I think this is a 3 — does it look like a 3?'
```

### Step 3: Visualize Manifolds
```bash
python visualize_conditional_ae.py
```
This creates:
- `conditional_ae_visualization.png` - Shows reconstruction using all 10 manifolds
- Console output showing manifold separation statistics

Look for:
- Green borders = predicted class manifold (should have low error)
- Blue borders = true class manifold
- High separation ratios (5-10x) indicate good class learning

### Step 4: Test Detection
```bash
python test_accuracy.py          # Automated testing
python detect.py                 # Interactive single image
python detect_batch.py           # Batch processing
python generate_report.py        # Visual report
```

## How It Works

### During Training
```
For each training sample:
  Image: 28×28 digit
  Label: 0-9
  ↓
  Label Embedding: 16-dim vector
  ↓
  Encoder: Concatenate(image, label_embedding) → latent
  ↓
  Decoder: Concatenate(latent, label_embedding) → reconstruction
  ↓
  Loss: MSE(reconstruction, original)
  
Result: Each digit learns its own reconstruction manifold
```

### During Inference
```
1. Classifier: "I think this is a 3" → prediction = 3
2. Autoencoder: Reconstruct using manifold-3
3. Check: reconstruction_error < threshold?
   - Low error → "Yes, looks like a 3" → ACCEPT
   - High error → "No, doesn't look like a 3" → REJECT
4. Mahalanobis: Is it close to class-3 prototype?
```

## Expected Results

### Manifold Separation (After Training)
```
Digit 0: Correct=0.003, Wrong=0.015, Ratio=5.0x
Digit 1: Correct=0.002, Wrong=0.012, Ratio=6.0x
...
Average ratio: 5-10x (higher is better)
```

### OOD Detection Improvement
- ✅ Better rejection of misclassified digits
- ✅ Better rejection of non-digits (letters, symbols)
- ✅ More interpretable: "Doesn't look like the predicted class"

## Visualizations

### conditional_ae_visualization.png
Shows for each test image:
1. Original image with true label
2. Classifier prediction and confidence
3. Reconstructions using ALL 10 class manifolds
4. Bar chart of reconstruction errors

**What to look for:**
- Predicted class (green) should have lowest error
- True class (blue) should also have low error (if correctly classified)
- Other classes should have noticeably higher errors
- Visual quality: correct class reconstruction should look sharper

## Architecture Details

### Model Parameters
- Label embedding dimension: 16
- Latent dimension: 64
- Total parameters: ~491K
- Input: (batch_size, 1, 28, 28) images + (batch_size,) labels
- Output: (batch_size, 1, 28, 28) reconstructions

### Training Configuration
- Optimizer: Adam (lr=1e-3)
- Loss: MSE (Mean Squared Error)
- Epochs: 5
- Device: CUDA (GPU)

## Troubleshooting

### Import Error: "No module named 'torch'"
Activate the conda environment:
```bash
conda activate pytorch
```

### Models not found
Train first:
```bash
python nn_train.py
```

### Low manifold separation
Try:
- Increase `embedding_dim` in MNISTAutoencoder
- Train for more epochs
- Increase `latent_dim` for more expressive manifolds

## Comparison: Before vs After

### Before (Standard Autoencoder)
- Learns: One global manifold for all digits
- Question: "Is this any digit?"
- Limitation: Hard to distinguish between digit classes

### After (Class-Conditional Autoencoder)
- Learns: 10 separate manifolds (one per digit)
- Question: "Is this specifically a 3?"
- Benefit: Clear separation between classes
- Biological: "I think this is a 3 — does it look like a 3?"

## References

See [CLASS_CONDITIONAL_IMPLEMENTATION.md](CLASS_CONDITIONAL_IMPLEMENTATION.md) for:
- Detailed implementation notes
- Theoretical background
- Comparison with related work (cVAE, cGAN)
- Advanced tuning options

## Next Steps

1. ✅ Architecture verified
2. ⏳ Train models: `python nn_train.py`
3. ⏳ Visualize: `python visualize_conditional_ae.py`
4. ⏳ Test: `python test_accuracy.py`
5. ⏳ Experiment with hyperparameters

---

**Status**: Ready to train! The architecture is implemented and tested.
