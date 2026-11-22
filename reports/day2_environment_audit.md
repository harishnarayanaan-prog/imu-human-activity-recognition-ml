# Day 2 Environment & Repository Audit Report

**Date & Time:** 2026-11-22  
**Project:** Human Activity Recognition from Wearable IMU Sensor Data using Time-Series Machine Learning  
**Repository Path:** `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml`  
**GitHub:** [https://github.com/harishnarayanaan-prog/imu-human-activity-recognition-ml](https://github.com/harishnarayanaan-prog/imu-human-activity-recognition-ml)  

---

## 1. Environment & Dependencies

| Tool / Package | Status | Detected Version | Purpose |
|:---|:---:|:---|:---|
| **Operating System** | Verified | Windows 11 (NT 10.0.26200.0, 64-bit) | Host platform |
| **Python** | Verified | `Python 3.14.3` (`C:\Python314\python.exe`) | Base interpreter |
| **Virtual Environment** | Verified | `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml\.venv` | Project-specific isolation |
| **Git** | Verified | `git version 2.56.1.windows.1` | Source control |
| **GitHub CLI (`gh`)** | Verified | `gh version 2.105.0` | GitHub repository automation |
| **numpy** | Verified | `2.5.5` | Tensor data structures |
| **pandas** | Verified | `3.1.1` | Tabular feature indexing |
| **scipy** | Verified | `1.18.1` | Signal processing, FFT, and spectral routines |
| **scikit-learn** | Verified | `1.9.1` | Classical ML pipelines, metrics, GroupShuffleSplit |
| **matplotlib** | Verified | `3.11.2` | Publication figures |
| **seaborn** | Verified | `0.13.2` | Confusion matrices & heatmaps |
| **joblib** | Verified | `1.6.0` | Model artifact persistence |
| **pytest** | Verified | `9.2.0` | Automated test runner |
| **PyTorch (`torch`)** | Verified | `2.15.0+cpu` | 1D-CNN deep sequence modeling |

---

## 2. Dataset Verification

The UCI Human Activity Recognition Using Smartphones dataset was audited from local disk:
- **Raw Extract Path:** `data/extracted/UCI HAR Dataset` (All 26 raw ASCII files intact)
- **Processed Artifacts Path:** `data/processed/`
- **Training Tensor:** `X_train.npy` shape `(7352, 128, 9)` — `float32`, 0 NaNs, 0 Infs
- **Testing Tensor:** `X_test.npy` shape `(2947, 128, 9)` — `float32`, 0 NaNs, 0 Infs
- **Labels:** 0-indexed integers `[0, 1, 2, 3, 4, 5]` corresponding to:
  - `0`: `WALKING`
  - `1`: `WALKING_UPSTAIRS`
  - `2`: `WALKING_DOWNSTAIRS`
  - `3`: `SITTING`
  - `4`: `STANDING`
  - `5`: `LAYING`
- **Participants:** 30 total subjects
  - 21 Training Subjects: `[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]`
  - 9 Testing Subjects: `[2, 4, 9, 10, 12, 13, 18, 20, 24]`
  - Subject Overlap: $\text{Train} \cap \text{Test} = \emptyset$ (Zero leakage)

---

## 3. Existing Test Suite Audit

Ran automated test suite (`pytest -v`):
- **Total Tests Collected:** 29 items across 6 test modules
- **Passed:** 29 (100%)
- **Failed:** 0
- **Modules Verified:** `test_download.py`, `test_loaders.py`, `test_validation.py`, `test_preprocessing.py`, `test_simulation.py`, `test_end_to_end_pipeline.py`.

---

## 4. Identified Technical Considerations for Day 2

1. **Test Set Isolation:** The 2,947 official test samples must remain quarantined during all feature engineering, scaler fitting, hyperparameter tuning, and model selection.
2. **Subject-Wise Cross-Validation:** Within the 21 training subjects, GroupShuffleSplit or GroupKFold must be used to partition subjects (e.g. 17 train subjects, 4 validation subjects) so validation mimics generalization to new individuals.
3. **Deep Learning Framework:** PyTorch 2.15.0+cpu is verified and functional on Python 3.14. It provides deterministic CPU training, clean 1D CNN modules (`nn.Conv1d`, `nn.BatchNorm1d`, `nn.Dropout`, `nn.AdaptiveAvgPool1d`, `nn.Linear`), and straightforward checkpoint serialization.
