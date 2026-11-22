# Data Leakage Prevention and Subject-Wise Splitting Methodology

**Project:** `imu-human-activity-recognition-ml`  
**Phase:** 3 – Data Splitting & Leakage Prevention  
**Author:** Harish Narayanaan  
**Date:** 2026-11-22  

---

## 1. The Risk of Data Leakage in Wearable Sensor ML

Wearable IMU sensor data presents unique data leakage challenges compared to traditional tabular or image datasets:

1. **Temporal Proximity and Window Overlap:**
   The UCI HAR dataset was collected continuously and sliced into fixed-width sliding windows of 128 samples (2.56 seconds) with **50% overlap** (64 samples). Consecutive windows share 1.28 seconds of identical inertial measurements.
2. **Subject Biometric Signatures (Idiosyncratic Biomechanics):**
   Every human subject exhibits unique gait dynamics, movement speeds, posture tilts, and sensor placement nuances. If windows from the same subject appear in both training and evaluation splits:
   - A machine learning model easily memorizes subject-specific biomechanical signatures (e.g. slight device tilt or individual walking cadence) rather than generalizable activity patterns.
   - Standard random train/test splits (e.g. `train_test_split(shuffle=True)`) produce artificially inflated test accuracies (>98–99%), which catastrophically fail in real-world deployment on unseen individuals.

---

## 2. Subject-Wise Separation Protocol

To establish true out-of-distribution generalizability across individuals, we enforce **strict Subject-Wise Group Partitioning**:

### A. Official Partitioning (Train vs. Held-Out Test)
- Total Subjects: **30 volunteers** (aged 19–48 years).
- **Official Training Partition (70%):** 21 subjects:  
  `[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]`  
  Total windows: **7,352**
- **Official Frozen Test Partition (30%):** 9 subjects:  
  `[2, 4, 9, 10, 12, 13, 18, 20, 24]`  
  Total windows: **2,947**
- **Subject Overlap between Official Train and Test:** **0.0% (Strictly Disjoint)**

### B. Validation Partitioning (Within Training Subjects Only)
To tune hyperparameters and perform model selection without touching the frozen test set, we partition the 21 training subjects into internal training and validation sets:
- **Validation Subject Selection Strategy:**
  - Ratio: ~20% of training subjects (4 or 5 subjects out of 21).
  - Validation Subjects: `[3, 11, 17, 27]` (4 subjects, representing 19.0% of training subjects and 1,395 windows).
  - Internal Training Subjects: 17 subjects (`[1, 5, 6, 7, 8, 14, 15, 16, 19, 21, 22, 23, 25, 26, 28, 29, 30]`, 5,957 windows).
  - Overlap: **0 subjects** ($\text{Train} \cap \text{Val} = \emptyset$).
- **Group K-Fold Cross-Validation:**
  - `SubjectWiseKFold` wraps `GroupKFold(n_splits=5)` using `subject_train` as group labels.
  - Ensures every validation fold contains exclusively unseen subjects.

---

## 3. Preprocessing and Feature Engineering Leakage Safeguards

In addition to window-level splitting, feature transformations must strictly prevent information flow from validation/test partitions into training:

1. **StandardScaler Fitting:**
   Any normalization scaler (such as `StandardScaler` in SVM or `ChannelWiseStandardScaler` for raw channels) is fitted **strictly on the internal training fold**. The validation and test features are strictly transformed using the parameters ($\mu, \sigma$) calculated from training data.
2. **Window-Local Feature Extraction:**
   Statistical features (mean, std, RMS, IQR, etc.) and frequency features (FFT dominant frequency, spectral energy, spectral entropy, centroid) are computed **locally within each individual 128-sample window**. They do not use inter-window statistics or future observations, preserving causal validity.
3. **Frozen Test Set Isolation:**
   The official test partition (`X_test.npy`, `X_test_features.npy`, `y_test.npy`) is kept completely untouched during feature selection, model architecture experimentation, and hyperparameter tuning. It is evaluated only once in Phase 8 for the final unbiased benchmark.

---

## 4. Verification Code Implementation

The protocol is implemented in `src/data/splitting.py`:
- `subject_wise_train_val_split()`: Generates disjoint train/val splits and verifies disjointness via `verify_disjointness()`.
- `SubjectWiseKFold`: Facilitates 5-fold cross-validation with subject grouping.
- Any attempt to introduce subject leakage raises an immediate `RuntimeError`.

---

## 5. Day 2.1 Methodology Audit and Preprocessing Leakage Correction

During the Day 2.1 audit, a critical vulnerability was identified and resolved:
- **Prior Vulnerability:** `ChannelWiseStandardScaler` had previously been fitted across the entire official training set (21 subjects) before the internal validation split was made. This allowed validation participants (`[3, 11, 17, 27]`) to leak statistical moments ($\mu, \sigma$) into the training representation.
- **Corrected Methodology:**
  1. Splitting is performed directly on **unscaled raw sensor windows** (`X_train_raw.npy`).
  2. The validation scaler is fitted **strictly on the 17 internal training subjects** (5,951 windows).
  3. Validation windows (1,401 windows) are normalized using the frozen internal training parameters.
  4. Features are extracted independently from the leak-free scaled partitions.
  5. The final test models are retrained on all 21 official training subjects with a scaler fitted on official training data only.
- For complete audit details and empirical parameter shifts, refer to `reports/methodology_correction_report.md`.
