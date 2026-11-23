# Data Leakage Audit and Methodology Correction Report

**Project:** `imu-human-activity-recognition-ml`
**Phase:** Day 2.1 – Data Leakage Audit, Methodology Correction & Final Test Independence Audit
**Author:** Harish Narayanaan
**Date:** 2026-11-22
**Status:** Audit Completed; Both Preprocessing & Test Leakages Corrected & Verified

---

## 1. Executive Summary & Audit Findings

An exhaustive independent machine learning methodology audit was performed on the data ingestion, preprocessing, splitting, feature extraction, and model training pipelines of `imu-human-activity-recognition-ml`.

The audit identified two distinct leakage vulnerabilities:
1. **Preprocessing Standardization Leakage (Validation Set):** `ChannelWiseStandardScaler` was originally fitted across all 21 official training subjects before performing the internal subject-wise validation split. Consequently, validation subjects (`[3, 11, 17, 27]`) contributed to the normalization statistics.
2. **Official Test-Set Evaluation Leakage (CNN Final Training):** During final model retraining, the CNN training loop called `cnn_final.fit(X_train_scaled, y_train, X_test_scaled, y_test)`. PyTorch's `fit()` routine evaluated test loss, reduced learning rates via `ReduceLROnPlateau`, and saved the "best" checkpoint according to minimum test loss (which occurred at epoch 2). This cherry-picked test performance and corrupted test independence.

Both vulnerabilities have been systematically eradicated.

---

## 2. Empirical Verification of Scaler Parameter Shift (Reconciled Authoritative Data)

To quantify the magnitude of the preprocessing contamination, channel statistics were recalculated directly from the actual unscaled official training array (`data/processed/X_train_raw.npy`, shape: `[7352, 128, 9]`) across:
- **A:** All 21 official training subjects (7,352 windows, 941,056 timesteps).
- **B:** The 17 internal training subjects (5,951 windows, 761,728 timesteps), strictly excluding validation subjects `[3, 11, 17, 27]`.

These values were programmatically generated and exported to `reports/authoritative_scaler_statistics.csv`:

$$\Delta \mu = \mu_{21} - \mu_{17}$$
$$\text{Rel } \sigma \text{ Shift} = \frac{\sigma_{17} - \sigma_{21}}{\sigma_{21}} \times 100\%$$

| Sensor Channel | 21-Subject Mean ($\mu_{21}$) | 17-Subject Mean ($\mu_{17}$) | Mean Difference ($\Delta \mu$) | 21-Subject Std ($\sigma_{21}$) | 17-Subject Std ($\sigma_{17}$) | Std Difference ($\Delta \sigma$) | Relative $\sigma$ Shift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `body_acc_x` | -0.000636 | -0.000677 | +0.000041 | 0.194846 | 0.196865 | +0.002019 | +1.04% |
| `body_acc_y` | -0.000292 | -0.000282 | -0.000010 | 0.122427 | 0.124796 | +0.002368 | +1.93% |
| `body_acc_z` | -0.000275 | -0.000292 | +0.000017 | 0.106879 | 0.110784 | +0.003906 | +3.65% |
| `body_gyro_x` | +0.000506 | +0.003275 | -0.002769 | 0.406815 | 0.413031 | +0.006216 | +1.53% |
| `body_gyro_y` | -0.000824 | -0.001715 | +0.000892 | 0.381854 | 0.403391 | +0.021537 | **+5.64%** |
| `body_gyro_z` | +0.000113 | -0.000269 | +0.000382 | 0.255743 | 0.264347 | +0.008604 | +3.36% |
| `total_acc_x` | +0.804749 | +0.800570 | **+0.004179** | 0.414112 | 0.420295 | +0.006183 | +1.49% |
| `total_acc_y` | +0.028755 | +0.028861 | -0.000106 | 0.390995 | 0.386745 | -0.004251 | -1.09% |
| `total_acc_z` | +0.086498 | +0.090496 | -0.003998 | 0.357769 | 0.364746 | +0.006977 | +1.95% |

For `body_gyro_y`, the standard deviation shifted by **+5.64%** and for `total_acc_x` the mean shifted by **+0.004179** when validation subjects were excluded. This proves that individual biometric variances noticeably altered the scaling parameters.

---

## 3. The Corrected Architecture and Data-Flow Topology

### Stage A: Validation Pipeline (Zero Leakage)
1. **Load Unscaled Raw Data:** Load raw, unscaled `X_train_raw.npy` (7,352, 128, 9), labels `y_train` (0-indexed), and subject IDs `subj_train`.
2. **Subject-Wise Split First:** Partition into:
   - Internal Training: 17 subjects (`[1, 5, 6, 7, 8, 14, 15, 16, 19, 21, 22, 23, 25, 26, 28, 29, 30]`, 5,951 windows).
   - Validation: 4 subjects (`[3, 11, 17, 27]`, 1,401 windows).
   - Overlap: **0.0% (strictly disjoint)**.
3. **Fit Scaler Strictly on Internal Training Partition:**
   `val_scaler = ChannelWiseStandardScaler()`
   `val_scaler.fit(X_train_internal_raw)` (fitted strictly on 5,951 windows).
4. **Transform Partitions:**
   `X_train_internal_scaled = val_scaler.transform(X_train_internal_raw)`
   `X_val_scaled = val_scaler.transform(X_val_raw)`
5. **Feature Extraction:**
   Extract 204 features independently:
   - `X_train_internal_feat = extractor.transform(X_train_internal_scaled)`
   - `X_val_feat = extractor.transform(X_val_scaled)`
6. **Model Training & Validation:**
   - **Random Forest:** Trained on `X_train_internal_feat`, evaluated on `X_val_feat` (Acc: 97.86%).
   - **SVM:** Trained on `X_train_internal_feat`, evaluated on `X_val_feat` (Acc: 96.29%).
   - **1D CNN:** Trained on `X_train_internal_scaled` with early stopping monitored on `X_val_scaled` (Acc: 99.07%, **best epoch = 32**).

### Stage B: Final Model Training & Official Test Evaluation (Zero Test Leakage)
1. **Fit Official Training Scaler:**
   `final_scaler = ChannelWiseStandardScaler()`
   `final_scaler.fit(X_train_raw)` (fitted across all 21 official training subjects).
2. **Transform Official Partitions:**
   `X_train_official_scaled = final_scaler.transform(X_train_raw)`
   `X_test_official_scaled = final_scaler.transform(X_test_raw)`
3. **Feature Extraction:**
   - `X_train_official_feat = extractor.transform(X_train_official_scaled)`
   - `X_test_official_feat = extractor.transform(X_test_official_scaled)`
4. **Retrain Final Models on Full Training Set:**
   - **Random Forest:** Retrained on all 7,352 feature rows.
   - **SVM:** Retrained on all 7,352 feature rows.
   - **1D CNN:** Implemented `CNN1DHAR.fit_fixed_epochs(X_train_scaled, y_train, epochs=32)`.
     - Initialized network from scratch with fixed seed.
     - Accepted strictly training arrays (rejects validation/test arguments).
     - Executed exactly 32 epochs (derived from validation Stage A).
     - Computed zero test loss and performed zero test-based checkpoint selection.
5. **Evaluate on Official Held-Out Test Partition:**
   Only after saving the final model, evaluate on the 2,947 official test windows:
   - 1D CNN: **93.08%** accuracy
   - SVM (RBF): **92.20%** accuracy
   - Random Forest: **91.38%** accuracy

---

## 4. Scientific Qualification

Because official test metrics were observed in Day 2, and subsequently audited in Day 2.1, this evaluation is transparently reported as a **methodology-corrected re-evaluation**, rather than an untouched first trial. All test leakage vectors (both preprocessing and model selection) have been definitively eliminated and covered with automated regression tests.
