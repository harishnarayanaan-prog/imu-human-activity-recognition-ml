# Preprocessing Methodology & Mathematical Formulation

## 1. Objective & Guardrails

The preprocessing pipeline converts heterogeneous windowed text files into unified, memory-efficient, leak-free numerical tensors optimized for time-series deep learning and machine learning models.

The primary engineering guardrails are:
1. **Zero Test Contamination:** Scalers and transformation parameters are fitted strictly on the training partition.
2. **Subject Independence:** The official train/test subject grouping is strictly preserved.
3. **Data Type Memory Optimization:** Sensor measurements are converted to `float32` for 50% memory reduction compared to raw text `float64`, with zero loss of numerical significance given sensor noise floors.
4. **Canonical Tensor Layout:** Structured as `(N, T, C)` = `(samples, timesteps, channels)` = `(N, 128, 9)`.

---

## 2. Mathematical Formulation of Channel-Wise Normalization

Wearable sensors record physical quantities with vastly disparate numerical scales:
- Body linear acceleration: $[-2\,g, +2\,g]$
- Total acceleration: $[0\,g, 6\,g]$ (due to static gravitational bias)
- Gyroscope angular velocity: $[-5\,\text{rad/s}, +5\,\text{rad/s}]$

To facilitate stable gradient descent in neural networks and uniform distance metrics in kernel methods, each of the $C=9$ channels is normalized independently using the empirical mean and standard deviation estimated across all time steps and windows in the **training set only**:

$$\mu_c = \frac{1}{N_{\text{train}} \cdot T} \sum_{i=1}^{N_{\text{train}}} \sum_{t=1}^{T} x_{i, t, c}$$

$$\sigma_c = \sqrt{\frac{1}{N_{\text{train}} \cdot T} \sum_{i=1}^{N_{\text{train}}} \sum_{t=1}^{T} (x_{i, t, c} - \mu_c)^2 + \epsilon}$$

Where:
- $N_{\text{train}} = 7,352$ windows
- $T = 128$ timesteps per window ($N_{\text{train}} \cdot T = 941,056$ observations per channel)
- $\epsilon = 10^{-8}$ prevents division by zero in zero-variance regimes

### Transformation Rule
For any sample $x_{i, t, c}$ in either training or test sets:

$$\hat{x}_{i, t, c} = \frac{x_{i, t, c} - \mu_c}{\sigma_c}$$

The exact parameters calculated from the training partition and stored in `data/processed/preprocessing_metadata.json` are:

| Channel | $\mu_c$ (Train Mean) | $\sigma_c$ (Train Std) |
|:---|:---:|:---:|
| `body_acc_x` | $+0.000100$ | $0.158356$ |
| `body_acc_y` | $-0.000780$ | $0.134015$ |
| `body_acc_z` | $-0.000570$ | $0.117260$ |
| `body_gyro_x` | $-0.001683$ | $0.322971$ |
| `body_gyro_y` | $-0.000673$ | $0.342410$ |
| `body_gyro_z` | $+0.000494$ | $0.230773$ |
| `total_acc_x` | $+0.785025$ | $0.311494$ |
| `total_acc_y` | $-0.138865$ | $0.245842$ |
| `total_acc_z` | $+0.071728$ | $0.267323$ |

---

## 3. Label Remapping Specification

To conform with PyTorch CrossEntropyLoss, TensorFlow SparseCategoricalCrossentropy, and scikit-learn convention, 1-indexed UCI labels are mapped to zero-indexed integers:

$$y_{\text{target}} = y_{\text{raw}} - 1$$

| Original UCI Label | Processed Integer Target | Class Name |
|:---:|:---:|:---|
| `1` | `0` | `WALKING` |
| `2` | `1` | `WALKING_UPSTAIRS` |
| `3` | `2` | `WALKING_DOWNSTAIRS` |
| `4` | `3` | `SITTING` |
| `5` | `4` | `STANDING` |
| `6` | `5` | `LAYING` |

---

## 4. Subject-Wise Cross-Validation Strategy

A major failure mode in wearable time-series machine learning is random window shuffling across train and test/validation folds. Because consecutive sliding windows overlap by 50%, adjacent windows share 64 time samples (1.28 seconds of identical sensor readings). Random shuffling results in acute temporal and identity data leakage, producing deceptively optimistic performance metrics.

To eliminate leakage:
1. **Primary Evaluation:** Evaluated on the 9 held-out subjects ($N=2,947$) who never participated in training.
2. **Validation Splits (`split_train_val_subject_wise`):** The 21 training subjects are partitioned by subject ID (e.g. 17 training subjects, 4 validation subjects), guaranteeing that validation folds measure generalization to unseen individuals.
