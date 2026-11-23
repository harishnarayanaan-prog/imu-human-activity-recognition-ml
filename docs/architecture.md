# System Architecture & Technical Design

**Project:** Human Activity Recognition from Wearable IMU Sensor Data
**Repository:** `https://github.com/harishnarayanaan-prog/imu-human-activity-recognition-ml`
**Storage Root:** `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml`
**Execution Environment:** Python 3.14.3 (CPU execution confirmed)

---

## 1. Architectural Philosophy

The **Human Activity Recognition from Wearable IMU Sensor Data using Time-Series Machine Learning** platform is engineered using production-grade software engineering and machine learning principles:
- **Modularity:** Separation of data ingestion, schema validation, signal transformation, feature engineering, modeling, and testing.
- **Data Integrity & Zero Test Leakage:** Strict subject-level partitioning ensuring no signal window or volunteer appears across both training and validation/test partitions.
- **Fixed-Epoch Final Training:** Elimination of test-set monitoring during final CNN training; duration derived exclusively from internal subject-wise validation experiments.
- **Reproducibility:** Seeded transformations, deterministic pre-processing parameters, and cryptographic SHA-256 auditing.
- **Defensive Engineering:** Explicit validation contracts asserting tensor dimensionalities, sampling rates, and absence of numerical anomalies ($\text{NaN}/\text{Inf}$).
- **Deployment Readiness:** Pre-configured support for ONNX graph export, ONNX Runtime evaluation, and micro-controller simulation.

---

## 2. End-to-End Pipeline Data Flow

```mermaid
flowchart TD
    subgraph S1["Phase 1: Ingestion & Validation"]
        A["Official UCI Archive (ZIP)"] -->|Stream Download & SHA-256 Checksum| B["data/raw/uci_har_dataset.zip"]
        B -->|Unzip & Structural Audit| C["data/extracted/UCI HAR Dataset"]
        C --> D["src/data/loaders.py"]
        D -->|Shape & Integrity Checks| E["src/data/preprocess.py"]
    end

    subgraph S2["Phase 2: Signal Preprocessing & Tensors"]
        E -->|Raw Signal Tensors (7352, 128, 9)| F["data/processed/X_train_raw.npy"]
        E -->|Raw Test Tensors (2947, 128, 9)| F_test["data/processed/X_test_raw.npy"]
        F --> G["ChannelWiseStandardScaler (Fitted on Train Only)"]
        G --> H["data/processed/X_train.npy & X_test.npy"]
    end

    subgraph S3["Phase 3: Feature Engineering (204 Features)"]
        H --> FE["src/features/feature_engineering.py"]
        FE -->|11 Time + 6 FFT per channel x 12 ch| MAT["data/processed/X_train_features.npy & X_test_features.npy"]
    end

    subgraph S4["Phase 4: Leak-Free Subject-Wise Validation"]
        F --> SPLIT["src/data/splitting.py"]
        SPLIT -->|Train (17 subjs: 5951 win)| TR_VAL["Internal Train"]
        SPLIT -->|Val (4 subjs: 1401 win)| HO_VAL["Internal Val: [3, 11, 17, 27]"]
        TR_VAL --> SCAL_VAL["Scaler fitted strictly on 17 subjects"]
        SCAL_VAL --> CNN_VAL["1D Deep CNN (Val Best Epoch = 32)"]
        TR_VAL --> RF_VAL["Random Forest (Val Acc: 97.86%)"]
        TR_VAL --> SVM_VAL["RBF SVM (Val Acc: 96.29%)"]
    end

    subgraph S5["Phase 5: Final Model Retraining & Frozen Test Set Evaluation"]
        H --> CNN_FIN["CNN1DHAR.fit_fixed_epochs(epochs=32)"]
        MAT --> RF_FIN["RandomForestHAR (Test Acc: 91.38%)"]
        MAT --> SVM_FIN["SVMHAR (Test Acc: 92.20%)"]
        CNN_FIN --> EVAL["Final Test Set (2947 windows, 9 unseen subjects)"]
        RF_FIN --> EVAL
        SVM_FIN --> EVAL
        EVAL --> RES["Test Evaluation Metrics (CNN Test Acc: 93.08%)"]
    end

    subgraph S6["Phase 6: Quality Assurance & Testing"]
        TESTS["pytest Test Suite (60 Passing Tests, 56% Coverage)"]
        TESTS --> REG["Leakage Regression Tests (tests/test_final_training_leakage.py)"]
    end
```

---

## 3. Component Specifications

### 3.1 Data Acquisition (`src/data/download_dataset.py`)
- Direct streaming from the official UCI repository (`https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip`).
- In-memory chunked streaming with SHA-256 verification (`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`).
- Automated zip integrity inspection via `testzip()` to prevent partial extraction.

### 3.2 Signal Loader (`src/data/loaders.py`)
- Stacks 9 discrete channel text files into contiguous 3D NumPy arrays:
  - Canonical shape: `(samples, timesteps, channels)` = `(N, 128, 9)`.
  - Canonical channel ordering:
    1. `body_acc_x`
    2. `body_acc_y`
    3. `body_acc_z`
    4. `body_gyro_x`
    5. `body_gyro_y`
    6. `body_gyro_z`
    7. `total_acc_x`
    8. `total_acc_y`
    9. `total_acc_z`

### 3.3 Subject-Wise Splitting & Leak-Free Scaler (`src/data/splitting.py`, `src/data/preprocess.py`)
- Internal validation: subjects `[3, 11, 17, 27]` reserved for validation ($1,401$ windows).
- Validation scaler fitted strictly on internal training subjects ($5,951$ windows).
- Full final training scaler fitted on all 21 official training subjects ($7,352$ windows).
- Authoritative statistical reconciliation documented in `reports/authoritative_scaler_statistics.csv`.

### 3.4 Feature Engineering (`src/features/feature_engineering.py`)
- 12 Channels total (9 raw + 3 derived magnitude channels: `body_acc_mag`, `body_gyro_mag`, `total_acc_mag`).
- 17 Features per channel ($12 \times 17 = 204$ features):
  - 11 Time-domain features (Mean, Std, Variance, Median, Min, Max, Range, IQR, Skewness, Kurtosis, RMS).
  - 6 Frequency-domain features via FFT (Spectral Energy, Entropy, Dominant Freq, Dominant Amp, Mean Freq, Spectral Spread).

### 3.5 Model Zoo & Fixed-Epoch Training (`src/models/`)
1. **Random Forest Classifier (`src/models/random_forest.py`):** 150 trees, max depth 25, Gini impurity.
2. **Support Vector Machine (`src/models/svm.py`):** RBF kernel, $C=10.0$, scale gamma.
3. **1D Deep CNN (`src/models/cnn_1d.py`):**
   - 3 Convolutional blocks with BatchNorm, ReLU, Dropout (0.3), and MaxPool.
   - Global Average Pooling + 2 Fully Connected layers with Dropout (0.5).
   - Parameters: **102,598**.
   - **`fit_fixed_epochs(X_train, y_train, epochs)`:** Guarantees zero test-set leakage by training strictly for fixed validation-derived epochs (32) without computing validation or test loss.

### 3.6 Synthetic IMU Generator (`src/simulation/synthetic_imu_generator.py`)
- Physics-based simulation engine for unit testing and pipeline stress-testing.
- Generates triaxial accelerometry and rate gyroscope waveforms with configurable cadence, harmonics, drift, and Gaussian noise.
- Decoupled from empirical training fixtures (`is_synthetic = True`).
