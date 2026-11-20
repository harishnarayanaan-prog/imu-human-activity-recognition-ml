# System Architecture & Technical Design

## 1. Architectural Philosophy

The **Human Activity Recognition from Wearable IMU Sensor Data using Time-Series Machine Learning** platform is engineered using production-grade software engineering and machine learning principles:
- **Modularity:** Separation of data ingestion, schema validation, signal transformation, and testing.
- **Data Integrity & Zero Test Leakage:** Strict subject-level partitioning ensuring no signal window or volunteer appears across both training and test partitions.
- **Reproducibility:** Seeded transformations, deterministic pre-processing parameters, and cryptographic SHA-256 auditing.
- **Defensive Engineering:** Explicit validation contracts asserting tensor dimensionalities, sampling rates, and absence of numerical anomalies ($\text{NaN}/\text{Inf}$).

---

## 2. End-to-End Pipeline Data Flow

```mermaid
flowchart TD
    subgraph S1["Phase 1: Ingestion & Validation"]
        A["Official UCI Archive (ZIP)"] -->|Stream Download & Checksum| B["data/raw/uci_har_dataset.zip"]
        B -->|Unzip & Structural Audit| C["data/extracted/UCI HAR Dataset"]
        C --> D["src/data/load_dataset.py"]
        D -->|Shape & Integrity Checks| E["src/data/validate_dataset.py"]
    end

    subgraph S2["Phase 2: Signal Preprocessing"]
        E -->|Raw Signal Tensors (7352, 128, 9)| F["src/data/preprocess.py"]
        F -->|Fit strictly on Train| G["ChannelWiseStandardScaler"]
        G -->|Transform Train & Test| H["data/processed/"]
        H --> H1["X_train.npy (32.3 MB)"]
        H --> H2["X_test.npy (13.0 MB)"]
        H --> H3["y_train.npy / y_test.npy"]
        H --> H4["subject_train.npy / subject_test.npy"]
        H --> H5["label_mapping.json"]
        H --> H6["preprocessing_metadata.json"]
    end

    subgraph S3["Phase 3: Exploration & Diagnostics"]
        D --> I["src/data/explore_dataset.py"]
        I --> J["reports/figures/*.png (7 Charts)"]
        I --> K["reports/dataset_exploration.md"]
        D --> L["notebooks/01_dataset_exploration.ipynb"]
    end

    subgraph S4["Phase 4: Simulation & Testing"]
        M["src/simulation/synthetic_imu_generator.py"] -->|Simulate IMU Physics| N["Synthetic Test Fixtures"]
        N --> O["pytest Test Suite (29 Tests)"]
        H --> O
    end
```

---

## 3. Component Specifications

### 3.1 Data Acquisition (`src/data/download_dataset.py`)
- Direct streaming from the official UCI repository (`https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip`).
- In-memory chunked streaming with automatic fallback to legacy UCI archives.
- Automated zip integrity inspection via `testzip()` to prevent partial extraction.

### 3.2 Signal Loader (`src/data/load_dataset.py`)
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
- Decouples tabular feature matrix (561 features) from raw temporal waveforms.

### 3.3 Data Validator (`src/data/validate_dataset.py`)
- Enforces strict contract assertions:
  - Shape bounds: Training `(7352, 128, 9)`, Testing `(2947, 128, 9)`.
  - Temporal integrity: 128 samples at 50 Hz = 2.56 seconds.
  - Zero tolerance for missing values (`NaN` or `Inf`).
  - Strict subject-wise disjointness: $\text{Subjects}_{\text{train}} \cap \text{Subjects}_{\text{test}} = \emptyset$.

### 3.4 Preprocessing Engine (`src/data/preprocess.py`)
- Remaps raw labels $1..6$ to standard zero-indexed targets $0..5$.
- Computes `ChannelWiseStandardScaler` parameters $(\mu_c, \sigma_c)$ **exclusively on the training set**.
- Applies learned parameters to transform test data without data snooping or leakage.
- Serializes high-throughput `.npy` binary arrays and structured metadata.

### 3.5 Synthetic IMU Engine (`src/simulation/synthetic_imu_generator.py`)
- Independent signal simulation module for software verification.
- Supports physical gait modeling (frequency harmonics, cadence, stance-phase impact) and stationary posture gravity projections.
- Configurable noise, drift, and bias parameters.
- Guaranteed separation: all synthetic outputs tagged with `is_synthetic = True` to prevent contamination of empirical benchmarks.
