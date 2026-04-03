# MNIST Digit Detection with Two-Stage OOD

**Academic Progression Framework:** This project implements a rigorous P vs ¬P (in-distribution vs out-of-distribution) detection system as an academic study, starting with MNIST digits as the training distribution (P) and testing the ability to reject non-MNIST inputs (¬P) without ever training on them. The goal is to perfect OOD detection methodology on MNIST, then demonstrate generalization to more complex datasets (Fashion-MNIST, CIFAR-10, medical imaging).

This project trains four complementary MNIST classifiers (CNN, FFN, NCT/Neocognitron, Fuzzy ART) and a class-conditional autoencoder (CCA), then runs detection through a two-stage OOD gate: 1) reconstruction error (Stage 1) and 2) Mahalanobis distance to class prototypes (Stage 2). Shared logic lives in src/first_ai/ to keep training and detection flows consistent.

## Project Architecture & Recent Improvements

**Separate Training Architecture:**
- Each model (FFN, CNN, NCT, ART) trains independently with dedicated training scripts
- Model-specific paths prevent conflicts: `MODEL_PATH_FFN`, `MODEL_PATH_CNN`, `MODEL_PATH_ART`
- OOD parameters computed separately for each classifier via `compute_ood_params.py`
- Shared utilities in `src/first_ai/` for dataloaders, logging, seeding, and OOD computation

**Fuzzy ART Enhancements:**
- Fixed category label assignment to prevent label drift during multi-pass training
- Reduced learning rate from 0.5 → 0.1 to prevent template convergence to universal attractors
- Category distribution analysis tracks digit-to-category mapping balance

**Key Design Principles:**
- Single source of truth for paths and hyperparameters in `config.py`
- Modular training allows independent experimentation per architecture
- Two-stage OOD detection (reconstruction + Mahalanobis) catches different failure modes

## Quick Start
- Dashboard (interactive): python dashboard.py
- Unified CLI (shared modules):
  - Train: python -m src.first_ai.cli train cnn --device auto --batch-size 256 --epochs 10
  - Detect single image: python -m src.first_ai.cli detect test_images/img_3.jpg
  - Batch detect: python -m src.first_ai.cli batch-detect test_images/
  - Report: python -m src.first_ai.cli report
- Legacy scripts remain runnable (e.g., python nn_train_cnn.py); they now call shared helpers.

## Layout (key paths)
- src/first_ai/ — shared package
  - data.py (MNIST loaders), train.py (early stopping trainer), ae_train.py, ood.py,
    artifacts.py, logging_utils.py, seeds.py, cli.py
- models/ — artifacts (classifiers, autoencoder, OOD params)
- outputs/ — reports/plots; captures/ — webcam saves; logs/ — log files
- Scripts: detect.py, detect_batch.py, generate_report.py, test_accuracy.py,
  test_class_thresholds.py, visualize_conditional_ae.py, camera.py, clean_project.py
- Training wrappers: nn_train_cnn.py, nn_train_ffn.py, nn_train_nct.py, nn_train_art.py
- Models: nn_model_cnn.py, nn_model_ffn.py, nn_model_nct.py, nn_model_art.py, autoencoder_model.py
- Config: config.py (devices, paths, hyperparameters)

## How detection works
- Stage 1: class-conditional autoencoder reconstruction error (reject obvious non-digits).
- Stage 2: Mahalanobis distance to class prototypes (per-class thresholds when available).
- Both stages share code in detection_utils.py; CLI and scripts call the same path to avoid drift.

## Train
- **Dashboard (recommended):** `python dashboard.py` → Option 1 (select FFN/CNN/NCT/ART/CCA)
- **Individual scripts:** `python nn_train_ffn.py` or `nn_train_cnn.py` or `nn_train_nct.py` or `nn_train_art.py`
- **CLI:** `python -m src.first_ai.cli train cnn --device auto --batch-size 256 --epochs 10`
- **After training:** Run Option 2 to compute OOD parameters for each model
- Artifacts land in `models/` with model-specific naming (e.g., `model_state_ffn.pth`, `ood_params_cnn.pth`)
- **Important:** Changing model architectures requires regenerating OOD params via `compute_ood_params.py`

## Detect & Report
- Single image: python -m src.first_ai.cli detect <image> or python detect.py.
- Batch: python -m src.first_ai.cli batch-detect <folder> or python detect_batch.py.
- Report: python -m src.first_ai.cli report or python generate_report.py.

## Utilities
- camera.py — capture and preprocess to MNIST format.
- visualize_conditional_ae.py — sanity-check AE manifolds vs classes.
- test_class_thresholds.py — inspect Mahalanobis thresholds and OOD behavior under noise.
- clean_project.py — remove generated artifacts while keeping data.

## Academic Goals & P vs ¬P Challenge

**Core Problem:** Train only on P (MNIST digits 0-9), then reliably detect ¬P (everything else) without ever seeing non-digit examples during training. This is the fundamental OOD detection challenge.

**Why This Matters:**
- Real-world systems encounter inputs outside training distribution
- No classifier is 100% accurate (humans can't perfectly distinguish ')' from '1')
- Goal: Maximize P recognition (~99%+) while rejecting ¬P (~90%+)

**Current Performance:**
- FFN: Best OOD detector (~60% combined accuracy on hard test cases)
- CNN: Strong classifier, moderate OOD performance
- ART: Classical ML baseline, complement-coded features

**Hard Test Cases:** Distinguishing handwritten '1' from ')' or '[' — visually similar but semantically different. Random images are trivial; lookalikes are the real challenge.

**Planned Progression:**
1. Perfect methodology on MNIST (current phase)
2. Validate on Fashion-MNIST (same 28×28 format, different domain)
3. Scale to CIFAR-10 (natural images, higher complexity)
4. Apply to real-world domains (medical imaging, anomaly detection)

## Notes on artifacts & consistency
- OOD detector (ood_detector.py) must match feature dims emitted by the classifier:
  - CNN/FFN: 128-dimensional embeddings
  - ART: 1568-dimensional complement-coded features (784 × 2)
- After changing architectures or hyperparameters, regenerate OOD params via dashboard Option 2
- Config paths in `config.py` point to model-specific files (no legacy `model_state.pth`)
- Mahalanobis distance thresholds calibrated at 90th, 95th, 99th percentiles per class
- Two-stage detection: Stage 1 (CCA reconstruction) catches obvious non-digits; Stage 2 (Mahalanobis) handles subtle cases

## Environment
- Python >= 3.8, PyTorch/torchvision per pyproject.toml or environment.yml.
- Set device via CLI --device auto|cpu|cuda; falls back to CPU when CUDA unavailable.

## Run on a Laptop (CPU) with Pre‑Trained Models
- Create env:
  - Conda: `conda env create -f environment.yml && conda activate pytorch`
  - Or pip: `pip install -e .` (reads pyproject.toml)
- Get trained artifacts onto the laptop:
  - Recommended: Git LFS (already configured). On the training machine:
    - `git lfs install`
    - `git add models/*.pth && git commit -m "Add trained model artifacts" && git push`
  - Then on the laptop: `git lfs install && git pull` (or fresh `git clone`).
  - Alternative: copy the `models/` folder manually once.
- Run detection:
  - Single image: `python detect.py test_images/img_3.jpg`
  - Dashboard: `python dashboard.py`
  - CLI: `python -m src.first_ai.cli detect test_images/img_3.jpg`
- Notes:
  - Inference uses `Config.DEVICE` and will run on CPU automatically if CUDA is unavailable.
  - Expected artifacts under `models/`: `model_state*.pth`, `autoencoder.pth`, `ood_params*.pth`.

## Testing
- python test_accuracy.py exercises the same detection path for a quick smoke test.
- Additional pytest scaffolding is planned under tests/ (see pyproject config).

## Author
David Cronin — January 2026
