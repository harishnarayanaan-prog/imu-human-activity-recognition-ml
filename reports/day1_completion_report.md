# Day 1 Completion & Acceptance Report

**Date & Time of Completion:** 2026-11-21
**Project:** Human Activity Recognition from Wearable IMU Sensor Data using Time-Series Machine Learning
**Repository Name:** `imu-human-activity-recognition-ml`
**Workspace Path:** `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml`
**Status:** **DAY 1 COMPLETE — ALL 20 ACCEPTANCE CRITERIA PASSED**

---

## 1. Environment Inspection Results

- **Operating System:** Windows 11 Enterprise / Pro (Microsoft Windows NT 10.0.26200.0, 64-bit)
- **Primary Python:** Python 3.14.3 (`C:\Python314\python.exe`)
- **Virtual Environment:** `.venv` created at `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml\.venv`
- **Pip In Venv:** Upgraded to `26.3`
- **Compiler:** GCC available (`D:\mingw64\bin\gcc.exe`)
- **Git Version:** `git version 2.56.1.windows.1`
- **Git User Config:** `harishnarayanaan-prog` (`harishnarayanaan@gmail.com`)
- **Disk Space Available:** `D:\` drive has 466.83 GB free out of 781.50 GB total.
- **Internet Connectivity:** Tested and verified live connection to `https://archive.ics.uci.edu` (HTTP 200).
- **Installed Packages & Versions (.venv):**
  - `numpy==2.5.5`
  - `pandas==3.1.1`
  - `scipy==1.18.1`
  - `scikit-learn==1.9.1`
  - `matplotlib==3.11.2`
  - `seaborn==0.13.2`
  - `requests==2.34.2`
  - `joblib==1.6.0`
  - `pytest==9.2.0`
  - `jupyter==1.1.1`

Documentation: [docs/environment_report.md](file:///D:/Projects/Wearable-Sensor-Platform/imu-human-activity-recognition-ml/docs/environment_report.md)

---

## 2. Dataset Download Source and Status

- **Dataset Name:** Human Activity Recognition Using Smartphones Dataset
- **Official Source:** UCI Machine Learning Repository
- **Direct Download URL:** `https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip`
- **DOI:** 10.24432/C54S4K
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Download Status:** Successfully downloaded 61,005,872 bytes archive to `data/raw/uci_har_dataset.zip`.
- **Integrity Validation:** Archive validated with `zipfile.testzip()` (0 corrupt members). Nested `UCI HAR Dataset.zip` unpacked to `data/extracted/UCI HAR Dataset`. All 26 required dataset files verified with non-zero byte sizes.

Documentation: [data/README.md](file:///D:/Projects/Wearable-Sensor-Platform/imu-human-activity-recognition-ml/data/README.md)

---

## 3. Dataset Statistics

- **Total Labelled Windows:** 10,299 windows
- **Official Training Set:** 7,352 windows (71.39%)
- **Official Test Set:** 2,947 windows (28.61%)
- **Window Length:** 128 discrete temporal readings (2.56 seconds at 50 Hz, 50% overlap)
- **Total Unique Participants:** 30 volunteers
  - Training Volunteers (21): `[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]`
  - Testing Volunteers (9): `[2, 4, 9, 10, 12, 13, 18, 20, 24]`
  - Subject Overlap: 0 (Strict subject-wise partition)
- **Activity Distribution:**
  - `LAYING`: 1,944 windows (18.88%)
  - `STANDING`: 1,906 windows (18.51%)
  - `SITTING`: 1,777 windows (17.25%)
  - `WALKING`: 1,722 windows (16.72%)
  - `WALKING_UPSTAIRS`: 1,544 windows (14.99%)
  - `WALKING_DOWNSTAIRS`: 1,406 windows (13.65%)
  - Imbalance Ratio: 1.38:1 (Well-balanced)

Documentation: [reports/dataset_exploration.md](file:///D:/Projects/Wearable-Sensor-Platform/imu-human-activity-recognition-ml/reports/dataset_exploration.md)

---

## 4. Actual Tensor Shapes & Dimensional Validation

All 9 triaxial inertial channels were stacked into structured 3D tensors:
1. `body_acc_x`
2. `body_acc_y`
3. `body_acc_z`
4. `body_gyro_x`
5. `body_gyro_y`
6. `body_gyro_z`
7. `total_acc_x`
8. `total_acc_y`
9. `total_acc_z`

- **Training Tensor (`X_train`):** `(7352, 128, 9)` — float32, exactly 0 NaNs, 0 Infs.
- **Testing Tensor (`X_test`):** `(2947, 128, 9)` — float32, exactly 0 NaNs, 0 Infs.
- **Training Labels (`y_train`):** `(7352,)` — int64 (remapped to 0..5).
- **Testing Labels (`y_test`):** `(2947,)` — int64 (remapped to 0..5).
- **Training Subjects (`subject_train`):** `(7352,)` — int64.
- **Testing Subjects (`subject_test`):** `(2947,)` — int64.

---

## 5. Preprocessing Operations Completed

1. **Modular Ingestion:** Automated loading from extracted text files without manual interaction.
2. **Contract Assertions:** Automated shape, type, range, and missing-value validation.
3. **Label Remapping:** Converted 1-indexed labels (1..6) to standard 0-indexed ML targets (0..5).
4. **Zero-Leakage Channel Standardization:** `ChannelWiseStandardScaler` fitted strictly on `X_train`. Transformation applied to `X_train` and `X_test` using learned train statistics $(\mu_c, \sigma_c)$. No test data was exposed during fitting.
5. **Subject-Wise Cross-Validation Helper:** Implemented `split_train_val_subject_wise` ensuring validation folds preserve whole-subject boundaries.
6. **Binary Serialization:** Saved high-throughput `.npy` files and metadata schemas.

Documentation: [docs/preprocessing_methodology.md](file:///D:/Projects/Wearable-Sensor-Platform/imu-human-activity-recognition-ml/docs/preprocessing_methodology.md)

---

## 6. Generated Files

### Code & Pipelines
- `src/__init__.py`
- `src/data/__init__.py`
- `src/data/download_dataset.py`
- `src/data/load_dataset.py`
- `src/data/validate_dataset.py`
- `src/data/preprocess.py`
- `src/data/generate_metadata.py`
- `src/simulation/__init__.py`
- `src/simulation/synthetic_imu_generator.py`

### Processed Datasets
- `data/processed/X_train.npy` (33,878,144 bytes, 32.31 MB, SHA-256: `65be85ca...`)
- `data/processed/y_train.npy` (58,944 bytes, SHA-256: `b9658087...`)
- `data/processed/subject_train.npy` (58,944 bytes, SHA-256: `66944a24...`)
- `data/processed/X_test.npy` (13,579,904 bytes, 12.95 MB, SHA-256: `4b40755b...`)
- `data/processed/y_test.npy` (23,704 bytes, SHA-256: `9d5716ef...`)
- `data/processed/subject_test.npy` (23,704 bytes, SHA-256: `533a077e...`)
- `data/processed/label_mapping.json` (464 bytes)
- `data/processed/preprocessing_metadata.json` (2,151 bytes)

### Notebooks & Documentation
- `notebooks/01_dataset_exploration.ipynb`
- `README.md`
- `data/README.md`
- `docs/architecture.md`
- `docs/data_dictionary.md`
- `docs/preprocessing_methodology.md`
- `docs/environment_report.md`
- `.gitignore`
- `pytest.ini`
- `requirements.txt`
- `requirements-dev.txt`

---

## 7. Figures Generated

All 7 publication-grade figures saved in `reports/figures/`:
1. `activity_distribution.png` (Class distribution across train vs test)
2. `participant_distribution.png` (Window count per participant across partitions)
3. `accelerometer_representative_activities.png` (Triaxial waveforms for walking, stairs, sitting, laying)
4. `gyroscope_representative_activities.png` (Triaxial angular velocity traces)
5. `stationary_vs_dynamic_comparison.png` (Direct side-by-side comparison of static vs dynamic signals)
6. `sensor_feature_correlation.png` (Pearson correlation heatmap across 9 channels)
7. `signal_amplitude_distributions.png` (Kernel density and histogram distributions)

---

## 8. Tests Executed & Results

- **Testing Framework:** `pytest 9.2.0`
- **Total Tests Executed:** 29 tests
- **Tests Passed:** 29
- **Tests Failed:** 0
- **Duration:** 4.59 seconds

### Test Breakdown by Module
- `tests/test_download.py`: 6 tests (Archive validity, zero-byte, corrupt files, missing extraction handling)
- `tests/test_loaders.py`: 5 tests (Label loading, feature naming, tensor shape validation, missing paths)
- `tests/test_validation.py`: 5 tests (Dimensional checks, NaN/Inf injection detection, invalid dtypes, subject leakage)
- `tests/test_preprocessing.py`: 5 tests (Scaler train-fitting, test non-leakage, inverse transforms, 0-indexing, subject-wise splits)
- `tests/test_simulation.py`: 5 tests (Config parameters, stationary generation, dynamic generation, error handling, batch datasets)
- `tests/test_end_to_end_pipeline.py`: 3 tests (Processed disk artifacts, exact tensor shapes, metadata parsing)

---

## 9. Errors Encountered & Resolutions

1. **PowerShell `powershell -Command` Nested Execution Issue:**
   - *Error:* Subshell parameter parsing conflict when invoking nested powershell.
   - *Resolution:* Executed PowerShell commands directly in the native PowerShell environment.
2. **Access Denied on `C:\Projects`:**
   - *Error:* Sandbox isolation restricts write permissions on root C: drive; workspace root is on `D:\Projects`.
   - *Resolution:* Created and executed workspace under `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml`, which possesses full read/write permissions.
3. **`ModuleNotFoundError: No module named 'src'` during direct script execution:**
   - *Error:* Python script executed with working directory outside repository root or without PYTHONPATH set.
   - *Resolution:* Added runtime path injection (`sys.path.insert(0, str(ROOT_DIR))`) in all entry point scripts and configured `pythonpath = .` in `pytest.ini`.
4. **Matplotlib Default Cache Path Permission Warning:**
   - *Error:* `C:\Users\LENOVO\.matplotlib` not writable in standard sandbox mode.
   - *Resolution:* Set `os.environ["MPLCONFIGDIR"]` to local project directory cache `.matplotlib_cache/`.

---

## 10. Remaining Limitations

1. **Pre-windowed & Pre-filtered Data:** Dataset consists of pre-segmented 128-sample windows with 50% overlap pre-filtered via 0.3 Hz Butterworth filters. Raw unsegmented ADC streams are not available from the source.
2. **Waist Placement Solely:** IMU was attached exclusively at the waist; models trained on this data will require domain adaptation if deployed on smartwatches or wristbands.
3. **Controlled Experimental Setting:** Scripted activity sequences in laboratory conditions do not reflect spontaneous real-world transitions.

---

## 11. Day 2 Readiness Assessment

The foundational data engineering, ingestion, validation, and zero-leakage preprocessing pipeline is **100% complete, verified, and ready for Day 2**.

**Immediate Day 2 Next Steps:**
1. **Time-Series Feature Engineering:** Compute statistical, temporal, and spectral features (mean, std, energy, entropy, autoregressive coefficients, FFT peak frequencies, wavelet decompositions).
2. **Classical Machine Learning Baselines:** Train and benchmark Random Forest, Extra Trees, Support Vector Classifiers, and Gradient Boosted Trees under subject-wise cross-validation.
3. **Deep Learning Sequence Modeling:** Implement 1D Convolutional Neural Networks (1D-CNN) and Bidirectional Long Short-Term Memory (Bi-LSTM) networks directly on `(N, 128, 9)` tensors.
4. **Comprehensive Evaluation Suite:** Generate per-class classification reports, normalized confusion matrices, and ROC-AUC curves.
