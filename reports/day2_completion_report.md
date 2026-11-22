# Day 2 Completion Report: Model Development and Evaluation

**Project Title:** Human Activity Recognition from Wearable IMU Sensor Data using Time-Series Machine Learning  
**Repository Name:** `imu-human-activity-recognition-ml`  
**GitHub:** `https://github.com/harishnarayanaan-prog/imu-human-activity-recognition-ml`  
**Author:** Harish Narayanaan  
**Date:** 2026-11-22  
**Status:** Completed & Validated (44/44 Tests Passing)  

---

## 1. Executive Summary

Day 2 ("Model Development and Evaluation") has been comprehensively executed, tested, and validated.

All required models—**Random Forest Classifier**, **Support Vector Machine (RBF Kernel)**, and **1D Deep Convolutional Neural Network (PyTorch)**—were implemented, trained under strict subject-wise data leakage safeguards, benchmarked on validation data, and ultimately evaluated against the official frozen test set.

Key achievements include:
- **Feature Engineering Engine:** High-performance vectorized extractor (`src/features/feature_engineering.py`) generating 204 features per window across 12 channels (9 raw + 3 derived magnitudes) in under 3 seconds with zero NaNs.
- **Subject-Wise Leakage Prevention:** Built `src/data/splitting.py` guaranteeing 0% subject overlap between train and validation partitions.
- **State-of-the-Art Deep Learning Model:** Developed a 3-stage 1D CNN with Batch Normalization, Max Pooling, and Global Average Pooling in PyTorch (102,598 parameters, 0.40 MB footprint).
- **Frozen Test Evaluation:** The 1D CNN achieved **93.15% Test Accuracy** and **0.9326 Macro F1** on the official 9-subject held-out test partition (2,947 windows), surpassing the original ESANN 2013 benchmark (89.3%).
- **Automated Test Suite:** Expanded test suite from 29 to **44 comprehensive automated tests**, with 100% passing rate (`pytest -q` in 11.97s).

---

## 2. Experimental Results Summary

### A. Subject-Wise Validation Benchmark (1,401 Windows, 4 Unseen Subjects)

| Model Architecture | Input Features | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Inference Latency | Model Size | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D Deep CNN** | Raw 9-Ch Waveforms | **98.93%** | **0.9910** | **0.9893** | **0.9899** | **0.9893** | 0.150 ms | **0.40 MB** | 54.24 s |
| **Random Forest** | 204 Engineered Feats | **97.72%** | 0.9787 | 0.9753 | 0.9764 | 0.9771 | **0.040 ms** | 6.06 MB | 6.24 s |
| **SVM (RBF Kernel)** | 204 Scaled Feats | **96.36%** | 0.9659 | 0.9607 | 0.9620 | 0.9637 | 0.280 ms | 2.27 MB | **2.92 s** |

### B. Final Frozen Test-Set Benchmark (2,947 Windows, 9 Unseen Subjects)

| Model Architecture | Input Features | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Test Latency | Model Size | Trainable Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D Deep CNN** | Raw 9-Ch Waveforms | **93.15%** | **0.9329** | **0.9325** | **0.9326** | **0.9312** | 0.173 ms | **0.40 MB** | 102,598 |
| **SVM (RBF Kernel)** | 204 Scaled Feats | **92.16%** | 0.9226 | 0.9207 | 0.9212 | 0.9214 | 0.266 ms | 2.27 MB | Non-parametric |
| **Random Forest** | 204 Engineered Feats | **91.62%** | 0.9171 | 0.9140 | 0.9149 | 0.9159 | **0.035 ms** | 6.06 MB | Non-parametric |

---

## 3. Detailed Phase Breakdown & Deliverables

### Phase 1 – Environment and Repository Audit
- Environment verified: Windows x86_64, Python 3.14.3.
- Installed `torch==2.15.0+cpu` inside project `.venv` without conflicts.
- Audit report documented at `reports/day2_environment_audit.md`.

### Phase 2 – Feature Engineering
- Created `src/features/feature_engineering.py`:
  - 9 base IMU channels (`body_acc`, `body_gyro`, `total_acc`) + 3 derived magnitudes (`body_acc_mag`, `body_gyro_mag`, `total_acc_mag`).
  - 13 statistical features (mean, std, var, min, max, median, range, RMS, MAV, IQR, energy, skewness, kurtosis).
  - 4 frequency features via 50 Hz RFFT (dominant frequency, spectral energy, spectral entropy, frequency centroid).
  - Extracted `(7352, 204)` train and `(2947, 204)` test feature matrices.
- Documented in `reports/feature_engineering_report.md` and `notebooks/02_feature_engineering.ipynb`.

### Phase 3 – Subject-Wise Splitting & Data Leakage Prevention
- Created `src/data/splitting.py` (`subject_wise_train_val_split`, `SubjectWiseKFold`).
- Zero subject overlap enforced ($\text{Train} \cap \text{Val} = \emptyset$).
- Documented in `reports/data_leakage_prevention.md`.

### Phase 4 – Random Forest Classifier
- Created `src/models/random_forest.py`.
- Trained 150-tree ensemble with MDI feature importance extraction.
- Validation Accuracy: **97.72%**, Test Accuracy: **91.62%**, Latency: **0.035 ms**.
- Model saved to `models/random_forest.joblib`.
- Top 20 features plotted in `reports/figures/rf_feature_importance.png`.

### Phase 5 – Support Vector Machine (SVM) Classifier
- Created `src/models/svm.py`.
- Built leak-free pipeline with `StandardScaler` fitted strictly on train.
- Validation Accuracy: **96.36%**, Test Accuracy: **92.16%**, Latency: **0.266 ms**.
- Model saved to `models/svm.joblib`.

### Phase 6 – 1D Deep CNN (PyTorch)
- Created `src/models/cnn_1d.py`.
- 3 Conv1D stages + BatchNorm1d + MaxPool1d + Global Average Pooling + Dropout + Linear.
- Trained end-to-end with AdamW and Early Stopping on validation loss.
- Validation Accuracy: **98.93%**, Test Accuracy: **93.15%**, Latency: **0.173 ms**, File size: **0.40 MB**.
- Model checkpoint saved to `models/cnn_1d.pt`.

### Phase 7 – Model Evaluation & Validation Comparison
- Created `src/evaluation/metrics.py`.
- Generated normalized confusion matrices for all models in `reports/figures/`.
- Exported comparative benchmark table `reports/model_comparison.csv`.
- Generated visual bar charts for Accuracy, Macro F1, Training Time, and Latency.
- Documented in `reports/validation_evaluation_report.md`.

### Phase 8 – Final Frozen Test-Set Evaluation
- Executed `scripts/evaluate_final_test.py` on untouched 2,947 test windows.
- Exported `reports/test_model_comparison.csv`.
- Generated final test confusion matrices:
  - `reports/figures/test_rf_confusion_matrix.png`
  - `reports/figures/test_svm_confusion_matrix.png`
  - `reports/figures/test_cnn_confusion_matrix.png`
- Documented in `reports/final_test_evaluation.md`.

### Phase 9 – Automated Testing Expansion
- Created comprehensive test modules:
  - `tests/test_feature_engineering.py` (6 tests)
  - `tests/test_splitting.py` (3 tests)
  - `tests/test_models.py` (4 tests)
  - `tests/test_evaluation.py` (2 tests)
- **Total test suite: 44 tests, 44 passed (0 failures) in 11.97s.**

### Phase 10 – Documentation
- Created `docs/model_architecture.md`.
- Created `docs/experimental_methodology.md`.
- Updated `README.md` with Day 2 benchmarks, architecture diagrams, and test results.

---

## 4. Repository Manifest of Day 2 Deliverables

```
imu-human-activity-recognition-ml/
├── docs/
│   ├── model_architecture.md               # Complete architecture specifications
│   └── experimental_methodology.md         # Experimental methodology & evaluation protocol
├── src/
│   ├── data/
│   │   └── splitting.py                    # Subject-wise train/val & K-fold split logic
│   ├── features/
│   │   ├── __init__.py                     # Feature engineering exports
│   │   └── feature_engineering.py          # 204-feature vectorized extraction engine
│   ├── models/
│   │   ├── __init__.py                     # Model exports (RF, SVM, CNN)
│   │   ├── random_forest.py                # Random Forest HAR wrapper & CLI
│   │   ├── svm.py                          # Support Vector Machine HAR pipeline & CLI
│   │   └── cnn_1d.py                       # PyTorch 1D Deep CNN & training harness
│   └── evaluation/
│       ├── __init__.py                     # Evaluation exports
│       └── metrics.py                      # Metrics, confusion matrix, & comparison runner
├── models/
│   ├── random_forest.joblib                # Serialized Random Forest model (6.06 MB)
│   ├── svm.joblib                          # Serialized SVM pipeline (2.27 MB)
│   └── cnn_1d.pt                           # PyTorch 1D CNN checkpoint (0.40 MB)
├── notebooks/
│   └── 02_feature_engineering.ipynb        # Interactive feature engineering walkthrough
├── reports/
│   ├── day2_environment_audit.md           # Day 2 environment verification report
│   ├── data_leakage_prevention.md          # Leakage prevention & splitting methodology
│   ├── feature_engineering_report.md       # Feature engineering report
│   ├── validation_evaluation_report.md     # Cross-model validation report
│   ├── final_test_evaluation.md            # Frozen test-set benchmark report
│   ├── model_comparison.csv                # Validation metrics CSV
│   ├── test_model_comparison.csv           # Final test metrics CSV
│   └── figures/
│       ├── rf_feature_importance.png       # Top 20 MDI feature importances
│       ├── rf_confusion_matrix.png         # RF validation confusion matrix
│       ├── svm_confusion_matrix.png        # SVM validation confusion matrix
│       ├── cnn_confusion_matrix.png        # CNN validation confusion matrix
│       ├── test_rf_confusion_matrix.png    # RF final test confusion matrix
│       ├── test_svm_confusion_matrix.png   # SVM final test confusion matrix
│       ├── test_cnn_confusion_matrix.png   # CNN final test confusion matrix
│       ├── model_accuracy_comparison.png   # Accuracy comparison bar chart
│       ├── macro_f1_comparison.png         # Macro F1 comparison bar chart
│       ├── training_time_comparison.png    # Training duration bar chart
│       └── inference_latency_comparison.png# Inference latency comparison bar chart
├── scripts/
│   └── evaluate_final_test.py              # Frozen test set benchmark runner
└── tests/
    ├── test_feature_engineering.py         # Feature extraction unit tests
    ├── test_splitting.py                   # Data splitting unit tests
    ├── test_models.py                      # Model inference & training unit tests
    └── test_evaluation.py                  # Metric & confusion matrix tests
```
