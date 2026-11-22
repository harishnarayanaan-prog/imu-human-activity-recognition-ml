# Data Leakage Audit and Methodology Correction Report

**Project:** `imu-human-activity-recognition-ml`  
**Phase:** Day 2.1 – Data Leakage Audit and Methodology Correction  
**Author:** Harish Narayanaan  
**Date:** 2026-11-22  
**Status:** Audit Completed; Methodology Corrected & Verified  

---

## 1. Executive Summary and Audit Findings

An independent ML methodology audit was performed on the data ingestion, preprocessing, splitting, feature extraction, and model training pipelines of `imu-human-activity-recognition-ml`.

### The Core Vulnerability: Preprocessing Standardization Leakage
In the original Day 1 and Day 2 implementation:
1. `src/data/preprocess.py` loaded the full official training partition (7,352 windows across all 21 training subjects).
2. It fitted `ChannelWiseStandardScaler` across **all 21 training participants** simultaneously:
   $$\mu_c = \frac{1}{N_{21} \cdot T}\sum_{i=1}^{N_{21}}\sum_{t=1}^T x_{i,t,c}, \quad \sigma_c = \sqrt{\frac{1}{N_{21} \cdot T}\sum_{i=1}^{N_{21}}\sum_{t=1}^T (x_{i,t,c} - \mu_c)^2}$$
3. The resulting normalized array was serialized as `data/processed/X_train.npy`.
4. In Day 2, feature extraction (`src/features/feature_engineering.py`) computed 204 features from `X_train.npy`.
5. Only *subsequently* did `subject_wise_train_val_split()` partition `X_train.npy` into internal training (17 subjects) and validation (4 subjects: `[3, 11, 17, 27]`).

### Impact of the Flaw
Because validation subjects `[3, 11, 17, 27]` were part of the 21 official training subjects:
- **Validation Biometric Contamination:** The individual mean kinetic offsets and dynamic variances of subjects 3, 11, 17, and 27 directly influenced the centering ($\mu_c$) and scaling ($\sigma_c$) of the internal training data.
- **Premature Normalization:** The validation windows were scaled using parameters calculated partly from themselves and partly from the training data before being held out.
- **Double Scaling in SVM:** SVM applied an additional `StandardScaler` inside its pipeline, but its input features were already derived from contaminated window representations.

---

## 2. Empirical Verification of Parameter Shift

To quantify the magnitude of this contamination, we compared the channel statistics computed across all 21 training subjects against those computed strictly across the 17 internal training subjects (excluding validation subjects 3, 11, 17, 27):

| Sensor Channel | All 21 Subjects Mean ($\mu_{21}$) | 17 Internal Subj Mean ($\mu_{17}$) | Absolute Mean $\Delta$ | All 21 Subjects Std ($\sigma_{21}$) | 17 Internal Subj Std ($\sigma_{17}$) | Absolute Std $\Delta$ | Relative $\sigma$ Shift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `body_acc_x` | -0.000636 | -0.000677 | 0.000041 | 0.194846 | 0.196865 | 0.002019 | +1.04% |
| `body_acc_y` | -0.000292 | -0.000282 | 0.000010 | 0.122427 | 0.124796 | 0.002368 | +1.93% |
| `body_acc_z` | -0.000275 | -0.000292 | 0.000017 | 0.106879 | 0.110784 | 0.003906 | +3.65% |
| `body_gyro_x`| 0.000506 | 0.003275 | **0.002769** | 0.406815 | 0.413031 | 0.006216 | +1.53% |
| `body_gyro_y`| -0.000824 | -0.001715 | 0.000892 | 0.381854 | 0.403391 | **0.021537** | **+5.64%** |
| `body_gyro_z`| 0.000113 | -0.000269 | 0.000382 | 0.255743 | 0.264347 | 0.008604 | +3.36% |
| `total_acc_x`| 0.804749 | 0.800570 | **0.004179** | 0.414112 | 0.420295 | 0.006183 | +1.49% |
| `total_acc_y`| 0.028755 | 0.028861 | 0.000106 | 0.390995 | 0.386745 | 0.004251 | -1.09% |
| `total_acc_z`| 0.086498 | 0.090496 | **0.003998** | 0.357769 | 0.364746 | 0.006977 | +1.95% |

For `body_gyro_y`, the standard deviation shifted by **5.64%** when validation subjects were excluded. This proves that validation participants had distinct angular velocity characteristics that distorted the training normalization.

---

## 3. The Corrected Architecture and Data-Flow Topology

To eliminate all leakage, we enforce a strict two-stage pipeline:

### Stage A: Validation Pipeline (Zero Leakage)
1. **Load Unscaled Raw Data:** Load raw, unscaled `X_train_raw` (7352, 128, 9), labels `y_train` (0-indexed), and subject IDs `subj_train`.
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
   - **Random Forest:** Trained on `X_train_internal_feat`, evaluated on `X_val_feat`.
   - **SVM:** `Pipeline([('scaler', StandardScaler()), ('svc', SVC())])` where `StandardScaler` fits only on `X_train_internal_feat`, then transforms `X_val_feat`.
   - **1D CNN:** Trained on `X_train_internal_scaled` with early stopping monitored on `X_val_scaled`.

### Stage B: Final Model Training & Official Test Evaluation
Once model architectures, features, and training regimes are validated:
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
   Retrain RF, SVM, and 1D CNN on the complete 7,352-sample official training partition.
5. **Evaluate on Official Held-Out Test Partition:**
   Evaluate the retrained models on the 2,947 official test windows (9 unseen subjects).

---

## 4. Scientific Qualification

Because official test metrics were already observed in Day 2, this evaluation is transparently reported as a **methodology-corrected re-evaluation**, rather than a completely unobserved blind test. No test performance figures were used to alter hyperparameters or model topologies.
