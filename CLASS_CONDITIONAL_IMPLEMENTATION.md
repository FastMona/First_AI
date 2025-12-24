# Class-Conditional Autoencoder Implementation Summary

## Overview

You asked for a class-conditional autoencoder implementation that mirrors biological perception: "I think this is a 3 — does it look like a 3?" This has been successfully implemented.

## What Changed

### 1. **autoencoder_model.py** - Core Architecture
**Before**: Standard autoencoder learning one global manifold
- Input: image only
- Output: reconstruction

**After**: Class-conditional autoencoder learning 10 separate manifolds
- Input: (image, label)
- Label embedding: 16-dimensional embedding for each class (0-9)
- Encoder: Concatenates flattened image (784) + label embedding (16) → latent (64)
- Decoder: Concatenates latent (64) + label embedding (16) → reconstruction (784)
- Output: class-specific reconstruction

**Key Methods**:
- `forward(x, labels)`: Reconstruct image conditioned on label
- `reconstruction_error(x, labels)`: Compute MSE using specific manifold
- `get_class_specific_error(x, predicted_class, return_all=False)`: 
  - Get error for predicted class
  - Optionally get errors for all 10 manifolds (useful for analysis)

### 2. **nn_train.py** - Training Updates
**Changes**:
- Now passes labels during training: `reconstruction = autoencoder(X, y)`
- Calibration uses **predicted labels** (not true labels) to simulate inference
- This is critical: "I think this is a 3, does it look like a 3?"
- Trains on all digits with their true labels to learn each manifold

### 3. **detection_utils.py** - Inference Updates
**Changes**:
- Classifier predicts first: `prediction = model(image)`
- Autoencoder reconstructs using predicted class manifold:
  ```python
  predicted_label = torch.tensor([prediction])
  recon_error = autoencoder.reconstruction_error(image, predicted_label)
  ```
- Sequential flow matches biological perception:
  1. "I think this is a 3" (classifier)
  2. "Does it look like a 3?" (autoencoder with class-3 manifold)
  3. "Is it close to the 3 prototype?" (Mahalanobis distance)

### 4. **visualize_conditional_ae.py** - New Visualization Tool
Demonstrates how the class-conditional autoencoder works:

**Visualization 1: Reconstruction Comparison**
- Shows original image
- Displays classifier prediction
- Shows reconstruction using ALL 10 manifolds side-by-side
- Color coding:
  - Green border = predicted class manifold
  - Blue border = true class manifold
- Bar chart of reconstruction errors for each manifold

**Visualization 2: Manifold Separation Analysis**
- For each digit class (0-9):
  - Compute mean error when using CORRECT manifold
  - Compute mean error when using WRONG manifolds
  - Calculate separation ratio (wrong/correct)
- Higher ratio = better manifold separation = better class learning

### 5. **README.md** - Updated Documentation
Added comprehensive sections:
- Explanation of class-conditional approach
- Biological perception analogy
- Benefits over standard autoencoders
- Usage instructions for visualization tool
- Manifold separation concept

## How It Works

### Training: Learning 10 Manifolds
```python
for image, label in training_data:
    # Learn reconstruction specific to this digit class
    reconstruction = autoencoder(image, label)
    loss = MSE(reconstruction, image)
    # Each class learns its own manifold geometry
```

### Inference: Class-Conditional Checking
```python
# Step 1: Classifier prediction
prediction = classifier(image)  # "I think this is a 3"

# Step 2: Check with predicted manifold
label_tensor = torch.tensor([prediction])
error = autoencoder.reconstruction_error(image, label_tensor)
# "Does it look like a 3?"

# Step 3: Compare to threshold
if error > threshold:
    reject()  # Doesn't look like the predicted digit
else:
    accept()  # Looks like the predicted digit
```

## Why This Is Better

### Standard Autoencoder (Old)
- Learns one manifold for ALL digits
- Reconstruction error measures "digit-ness" in general
- Hard to distinguish between digit classes
- Example: A "5" and a "3" might both reconstruct well

### Class-Conditional Autoencoder (New)
- Learns 10 separate manifolds (one per digit)
- Reconstruction error measures "looks like class k"
- Clear separation between classes
- Example: 
  - A "5" reconstructed with "5 manifold" → low error
  - A "5" reconstructed with "3 manifold" → high error
  - A letter "A" reconstructed with any manifold → very high error

### Biological Plausibility
Humans don't have a single "object recognition" system. We have:
1. **Hypothesis formation**: "I think this is a 3"
2. **Verification**: "Does it match my mental model of a 3?"
3. **Prototype matching**: "Is it similar to other 3's I've seen?"

The class-conditional autoencoder implements steps 1-2, and Mahalanobis distance implements step 3.

## Expected Results

### Manifold Separation Metrics
After training, you should see:
- **Correct manifold error**: ~0.001 - 0.005 (low)
- **Wrong manifold error**: ~0.01 - 0.05 (higher)
- **Separation ratio**: 3-10x (wrong/correct)
- Higher ratios indicate better class-specific learning

### OOD Detection
The two-stage gate now works better:
1. **Stage 1 (Autoencoder)**: Rejects non-digits and mismatched classes
   - Non-digit (letter "A") → high error with ALL manifolds
   - Misclassified digit → high error with wrong manifold
   
2. **Stage 2 (Mahalanobis)**: Rejects outliers within a class
   - Unusual "3" that still looks like a "3" → passes stage 1
   - But far from typical "3" prototype → rejected by stage 2

## Usage Instructions

### 1. Train the Models
```bash
python nn_train.py
```
This will train:
- CNN classifier
- Class-conditional autoencoder (5 epochs)
- Mahalanobis OOD detector

Output files:
- `model_state.pth` - Classifier
- `autoencoder.pth` - Conditional autoencoder + threshold
- `ood_params.pth` - Mahalanobis parameters

### 2. Visualize Manifolds
```bash
python visualize_conditional_ae.py
```
This creates:
- `conditional_ae_visualization.png` - Visual comparison
- Console output with manifold separation statistics

Look for:
- Low error for correct class (green border)
- High error for wrong classes
- Clear visual differences in reconstructions

### 3. Test Detection
```bash
python test_accuracy.py
python detect.py  # Interactive single image
python detect_batch.py  # Batch processing
```

The detection now uses class-conditional reconstruction:
- Classifier predicts class k
- Autoencoder checks with manifold k
- Better rejection of OOD samples

## Implementation Notes

### Label Embedding
- Each class (0-9) gets a learnable 16-dimensional embedding
- This embedding is concatenated with image features
- Allows the network to learn class-specific transformations
- Similar to conditional GANs and conditional VAEs

### Training Strategy
- Train on (image, true_label) pairs
- This ensures each manifold learns correctly
- Calibration uses predicted labels to match inference conditions

### Backward Compatibility
The changes are designed to be minimal:
- Main API change: `autoencoder(image, labels)` instead of `autoencoder(image)`
- Old models won't work - need to retrain
- All other code (detection_utils, test scripts) updated accordingly

## Next Steps

1. **Train and visualize**: Run training then visualization to see manifold separation
2. **Compare performance**: Test on OOD samples to see if rejection rate improves
3. **Tune hyperparameters**: 
   - Increase `embedding_dim` for more expressive label embeddings
   - Increase `latent_dim` for more complex manifolds
   - Train for more epochs for better convergence
4. **Advanced analysis**:
   - Visualize label embeddings (t-SNE)
   - Compare reconstruction across digit pairs (which are most confused?)
   - Analyze OOD detection on challenging samples

## References

This approach is inspired by:
- **Conditional VAEs** (Sohn et al., 2015)
- **Conditional GANs** (Mirza & Osindero, 2014)
- **Biological perception models** in cognitive neuroscience
- The concept of "analysis-by-synthesis" in vision research

The key insight: Don't just learn to reconstruct — learn to reconstruct **conditioned on what you think you're seeing**.
