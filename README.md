# jy-anomaly-ae

Autoencoder-based anomaly detection pipeline for toroidal current density ((j_y)) plots generated from MHD simulations.

---

## 1. Motivation

Magnetohydrodynamic (MHD) simulations (e.g., M3D-C1) are computationally expensive and can sometimes produce **invalid or low-quality outputs** due to:

* Mesh failures
* Solver non-convergence
* Physically inconsistent plasma profiles

These failures are not always obvious numerically, but they are **visually detectable** in (j_y) plots.

If such corrupted outputs are used downstream (e.g., for training surrogate models like Gaussian Process Regression), they can **degrade model quality significantly**.

### Goal

Build an automated system that:

* Learns what *valid plasma outputs* look like
* Detects anomalous simulation outputs
* Filters bad data before it enters downstream pipelines

---

## 2. Why Autoencoders?

This project uses a **convolutional autoencoder (AE)** for anomaly detection.

### Core Idea

Train the AE **only on healthy (normal) images**.

* The model learns a compressed representation of normal (j_y) structure
* It reconstructs normal images well
* It fails to reconstruct anomalous images

### Anomaly Signal

We measure:

[ \text{Reconstruction Error} = | x - \hat{x} |^2 ]

* Low error → looks like training data → **healthy**
* High error → deviates from learned structure → **anomaly**

This makes the AE a **one-class anomaly detector**.

---

## 3. Anomaly Types

The dataset contains four classes:

| Class            | Description               | Frequency Characteristics |
| ---------------- | ------------------------- | ------------------------- |
| healthy          | Physically correct plasma | baseline                  |
| bad_black_core   | Hollow/void at core       | low-frequency anomaly     |
| bad_nonconverged | Solver instability        | mixed/noisy               |
| broken_lcfs      | Edge mesh failure (LCFS)  | high-frequency            |

Each class corresponds to a **distinct physical failure mode**.

---

## 4. Pipeline Overview

```
M3D-C1 Simulation
        ↓
  j_y Plot Image
        ↓
 Autoencoder
        ↓
Reconstruction Error
        ↓
Anomaly Score
        ↓
Filter / Flag
```

Only **healthy outputs** are passed to downstream models.

---

## 5. Model Architecture

The autoencoder is a **configurable CNN with a flat bottleneck**:

* Encoder: Conv → (optional BN) → ReLU → MaxPool
* Flatten → Dense → **latent vector (bottleneck)**
* Dense → reshape
* Decoder: Upsample → Conv blocks
* Output: Sigmoid

### Key Design Choices

* **Flat bottleneck (vector)** instead of spatial bottleneck
* **Nearest-neighbor upsampling + conv** (avoids checkerboard artifacts)
* Optional **BatchNorm**
* Symmetric spatial scaling
* Asymmetric depth allowed (encoder ≠ decoder)

---

## 6. Hyperparameter Optimization

We use **Optuna** to search over architecture and training parameters.

### Search Space (10D)

* use_hpf (high-pass filter)
* hpf_sigma
* bottleneck_dim
* n_enc_layers
* n_dec_layers
* base_channels
* growth_factor
* use_batchnorm
* learning rate
* batch size

### Objective Function

Per-class AUC on validation set:

* Hard constraint: key classes must exceed threshold
* Weighted objective prioritizes difficult anomalies (broken_lcfs)

---

## 7. Repository Structure

```
ae_lib/        → model, data loading, training, evaluation

tune_lib/      → Optuna search logic

configs/       → experiment configs

train.py       → train a single model

tune.py        → run hyperparameter search

requirements.txt
```

---

## 8. How to Run

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Train a model

```bash
python train.py --config configs/tests/test_run.yaml --gpu 0
```

### 3. Run hyperparameter search

```bash
python tune.py
```

---

## 9. Hardware Notes

Designed for GPU acceleration (tested on AMD MI210 with ROCm).

* Uses PyTorch ROCm backend
* Requires proper GPU visibility configuration

---

## 10. Important Notes

* The model is trained **only on healthy data**
* Test data must remain **completely isolated**
* Evaluation is done using **per-class AUC**, not pooled metrics

---

## 11. Use Case

This model serves as a **data quality filter** in a larger pipeline:

```
Simulation → AE Filter → Clean Dataset → Surrogate Model (GPR)
```

---

## 12. Future Work

* Improve detection of edge anomalies (broken_lcfs)
* Explore alternative decoders (e.g., ConvTranspose)
* Incorporate multi-scale feature learning

---

## 13. Author

Jiya Rathi
M.S. Computer Engineering, SDSU
