# Human Activity Recognition from Wearable IMU Sensor Data using Time-Series Machine Learning

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests: 77 Passed](https://img.shields.io/badge/tests-77%20passed-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 1. Project Overview & Problem Statement

Wearable inertial measurement units (IMUs)—comprising triaxial accelerometers and gyroscopes—are ubiquitous across consumer smartwatches, fitness trackers, and clinical rehabilitation monitors. Accurately recognizing human physical activities (locomotion vs. stationary postures) from continuous inertial sensor streams is essential for health monitoring, fall detection, sports performance analytics, and context-aware ubiquitous computing.

However, wearable time-series machine learning introduces distinctive engineering challenges:
1. **Temporal & Kinematic High Variance:** Multi-channel kinematic measurements (linear acceleration and angular velocity) have vastly different physical units ($g$ vs. $\text{rad/s}$) and dynamic ranges.
2. **Data Leakage Vulnerability:** Windowed sensor streams with 50% overlap share identical motion samples between adjacent windows. Random splitting leads to severe information leakage and overly optimistic, ungeneralizable models.
3. **Subject Variability:** Biomechanical gait signatures vary across individuals due to stature, cadence, and habits. Models must be rigorously evaluated on unseen subjects.

This project implements a robust, reproducible end-to-end Machine Learning pipeline that processes windowed triaxial IMU sensor recordings, enforces strict subject-wise partitioning, normalizes sensor channels without data snooping, and establishes validated foundations for deep neural networks and baseline classifiers.

---

## 2. Project Objectives

- **Automated Data Ingestion:** Automate acquisition and SHA-256 verification of the official UCI Human Activity Recognition (HAR) dataset.
- **Robust Multi-Channel Tensor Pipeline:** Ingest, parse, and stack 9 triaxial inertial channels into consistent `(samples, timesteps, channels)` 3D tensors: `(7352, 128, 9)` train and `(2947, 128, 9)` test.
- **Strict Leakage Prevention:** Retain official subject partitions and enforce subject-disjoint validation routines where $\text{Train} \cap \text{Test} = \emptyset$.
- **Zero-Snooping Normalization:** Fit standardization parameters strictly on the training partition and broadcast learned coefficients to test partitions.
- **Synthetic Sensor Emulation:** Implement an independent, physically plausible synthetic IMU signal generator for pipeline testing, cleanly decoupled from empirical evaluations.
- **Comprehensive Quality Assurance:** Maintain a complete test suite with 100% passing unit and integration tests.

---

## 3. Dataset Source & Attribution

- **Dataset:** Human Activity Recognition Using Smartphones Dataset
- **Official Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **DOI:** [10.24432/C54S4K](https://doi.org/10.24432/C54S4K)
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

> **Citation:**
> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge L. Reyes-Ortiz.
> *A Public Domain Dataset for Human Activity Recognition Using Smartphones*.
> 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013.

### Sensor Modality Details
- **Subjects:** 30 volunteers (aged 19–48 years).
- **Device:** Samsung Galaxy S II mounted at waist level.
- **Sampling Rate:** 50 Hz (20 ms sample period).
- **Window Length:** 128 samples (2.56 seconds duration) with 50% overlap (64 samples step size).
- **Sensor Preprocessing Note:** **IMPORTANT:** The sensor recordings in this dataset are preprocessed, windowed inertial signals. They are **not** continuous, unsegmented raw sensor recordings.

---

## 4. System Architecture

```mermaid
flowchart LR
    A["Raw Data (UCI HAR)"] --> B["Ingestion & Validation"]
    B --> C["Tensor Stacking (N, 128, 9)"]
    C --> D["Train-Fitted StandardScaler"]
    D --> E["Processed NumPy Artifacts"]
    E --> F["Time-Series ML / Deep Learning"]
```

See [docs/architecture.md](docs/architecture.md) for detailed architectural blueprints.

---

## 5. Repository Structure

```
imu-human-activity-recognition-ml/
├── data/
│   ├── raw/                           # Downloaded official zip archive
│   ├── extracted/                     # Unpacked official UCI dataset
│   ├── processed/                     # Leak-free normalized NumPy arrays
│   │   ├── X_train.npy                # (7352, 128, 9) float32
│   │   ├── y_train.npy                # (7352,) int64 [0..5]
│   │   ├── subject_train.npy          # (7352,) int64 [1..30]
│   │   ├── X_test.npy                 # (2947, 128, 9) float32
│   │   ├── y_test.npy                 # (2947,) int64 [0..5]
│   │   ├── subject_test.npy           # (2947,) int64 [1..30]
│   │   ├── label_mapping.json         # Bi-directional label mappings
│   │   └── preprocessing_metadata.json# Pipeline configuration & scaler parameters
│   └── README.md                      # Comprehensive dataset documentation
├── docs/
│   ├── architecture.md                # Component design & pipeline diagram
│   ├── data_dictionary.md             # Channel definitions, units, & schema
│   ├── preprocessing_methodology.md   # Mathematical normalization & split logic
│   └── environment_report.md          # Platform & toolchain audit report
├── notebooks/
│   └── 01_dataset_exploration.ipynb   # Interactive analysis notebook
├── reports/
│   ├── dataset_exploration.md         # Genuine empirical observations
│   ├── day1_completion_report.md      # Acceptance criteria verification
│   └── figures/                       # Publication-grade visualizations
│       ├── activity_distribution.png
│       ├── participant_distribution.png
│       ├── accelerometer_representative_activities.png
│       ├── gyroscope_representative_activities.png
│       ├── stationary_vs_dynamic_comparison.png
│       ├── sensor_feature_correlation.png
│       └── signal_amplitude_distributions.png
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_dataset.py        # Automated download & checksum verification
│   │   ├── load_dataset.py            # Modular file & tensor loaders
│   │   ├── validate_dataset.py        # Dimensional & leakage contract checkers
│   │   ├── preprocess.py              # Zero-leakage standardization & remap
│   │   └── generate_metadata.py       # Artifact auditing & SHA-256 computation
│   └── simulation/
│       ├── __init__.py
│       └── synthetic_imu_generator.py # Physically plausible synthetic generator
├── tests/
│   ├── __init__.py
│   ├── test_download.py               # Downloader & archive validation tests
│   ├── test_loaders.py                # File reading & stacking tests
│   ├── test_validation.py             # Integrity & leakage detection tests
│   ├── test_preprocessing.py          # Scaler & split tests
│   ├── test_simulation.py             # Synthetic generator tests
│   └── test_end_to_end_pipeline.py    # Processed artifact integration tests
├── .gitignore                         # Excludes large binaries & cache
├── pytest.ini                         # Pytest configuration
├── requirements.txt                   # Production dependencies
└── requirements-dev.txt               # Testing & development dependencies
```

---

## 6. Environment Setup & Installation (Windows)

### Prerequisites
- Windows 10 / 11 (64-bit)
- Python 3.10 to 3.14
- Git for Windows

### Setup Virtual Environment
```powershell
# Navigate to workspace
cd imu-human-activity-recognition-ml

# Create project-specific virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements-dev.txt
```

---

## 7. Execution Guide

### Step 1: Download & Extract Dataset
```powershell
python src/data/download_dataset.py
```

### Step 2: Run Exploratory Analysis & Generate Figures
```powershell
python src/data/explore_dataset.py
```

### Step 3: Run Zero-Leakage Preprocessing
```powershell
python -m src.data.preprocess
```

### Step 4: Run Feature Engineering (204 Features)
```powershell
python -m src.features.feature_engineering
```

### Step 5: Execute Complete Methodology-Corrected Pipeline
```powershell
python scripts/run_corrected_pipeline.py
```
Or execute individual model components:
```powershell
# Train Random Forest Classifier (leak-free)
python -m src.models.random_forest

# Train Support Vector Machine (leak-free)
python -m src.models.svm

# Train 1D Deep CNN (PyTorch, leak-free)
python -m src.models.cnn_1d

# Run Validation Benchmark
python -m src.evaluation.metrics

# Run Final Frozen Test Set Evaluation
python scripts/evaluate_final_test.py
```

### Step 6: Export & Benchmark Optimized Edge Models (Day 3)
```powershell
# Export PyTorch CNN to ONNX FP32, FP16, and INT8 (Static QDQ & Dynamic)
python scripts/export_onnx.py

# Benchmark Latency, Throughput & Accuracy Across All Precision Variants
python scripts/benchmark_optimized_models.py

# Run Real-Time IMU Streaming Pipeline Emulation (Empirical & Synthetic)
python scripts/run_streaming_demo.py
```

### Step 7: Execute Complete Automated Test Suite (77 Tests)
```powershell
pytest -q
```

---

## 8. Methodology Audit & Preprocessing Leakage Correction (Day 2.1)

An ML methodology audit identified that in the initial Day 2 release, `ChannelWiseStandardScaler` was fitted across all 21 training subjects prior to subject-wise validation splitting, which allowed validation subjects (`[3, 11, 17, 27]`) to contribute to the normalization statistics ($\mu, \sigma$).

### The Corrected Data-Flow Protocol:
1. **Unscaled Splitting:** Subject-wise partitioning is performed directly on unscaled raw sensor windows (`X_train_raw.npy`).
2. **Internal-Only Fitting:** The validation scaler is fitted **strictly on the 17 internal training subjects** (5,951 windows).
3. **Frozen Transformation:** Validation windows (1,401 windows) are scaled using the frozen internal parameters.
4. **Final Model Retraining:** Final models are retrained on the full official training partition (7,352 samples) with a scaler fitted on official training data, then evaluated on the 2,947 official test windows.

Full audit documentation is available in [reports/methodology_correction_report.md](reports/methodology_correction_report.md) and historical metrics are preserved in [reports/historical_day2_results.md](reports/historical_day2_results.md).

---

## 9. Model Benchmark Performance (Corrected Methodology)

All reported metrics are genuine empirical values directly evaluated on physical CPU execution.

### A. Corrected Subject-Wise Validation Benchmark (1,401 Windows, 4 Unseen Subjects)
*Zero subject overlap; normalization scaler fitted strictly on internal training partition.*

| Model Architecture | Input Features | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Inference Latency | Model Size | Training Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D Deep CNN** | Raw 9-Ch Waveforms | **99.07%** | **0.9922** | **0.9909** | **0.9914** | **0.9907** | 0.198 ms / win | **0.40 MB** | 147.01 s |
| **Random Forest** | 204 Engineered Feats | **97.86%** | 0.9801 | 0.9761 | 0.9775 | 0.9785 | **0.025 ms / win** | 6.06 MB | 6.38 s |
| **SVM (RBF Kernel)** | 204 Scaled Feats | **96.29%** | 0.9653 | 0.9597 | 0.9611 | 0.9630 | 0.276 ms / win | 2.27 MB | **4.01 s** |

### B. Corrected Official Test-Set Benchmark (2,947 Windows, 9 Unseen Subjects)
*Methodology-corrected re-evaluation on complete 7,352 training samples, evaluated on held-out test partition.*

| Model Architecture | Input Modality | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Test Latency | Model Size | Trainable Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D Deep CNN** | Raw 9-Ch Waveforms | **93.08%** | **0.9304** | **0.9322** | **0.9308** | **0.9300** | 0.157 ms / win | **0.40 MB** | 102,598 |
| **SVM (RBF Kernel)** | 204 Scaled Feats | **92.20%** | 0.9230 | 0.9215 | 0.9217 | 0.9218 | 0.341 ms / win | 2.46 MB | Non-param |
| **Random Forest** | 204 Engineered Feats | **91.38%** | 0.9148 | 0.9113 | 0.9124 | 0.9136 | **0.025 ms / win** | 6.92 MB | Non-param |

---

## 10. Day 3: Model Optimization & Embedded Deployment Simulation

Day 3 evaluates model compression, precision reduction, and edge runtime feasibility by exporting the validated PyTorch 1D CNN baseline (`models/cnn_1d.pt`) into optimized ONNX graph representations and benchmarking against strict embedded constraints.

### A. Optimization & Edge Emulation Pipeline Architecture

```mermaid
flowchart TD
    subgraph Host ["Physical Host & Training Pipeline"]
        A["PyTorch Baseline (FP32)<br/>models/cnn_1d.pt (420.7 KB)"] --> B["ONNX FP32 Graph Export<br/>models/optimized/cnn_1d_fp32.onnx"]
        B --> C["FP16 Half-Precision Conversion<br/>models/optimized/cnn_1d_fp16.onnx"]
        B --> D["Static QDQ INT8 Calibration<br/>Strictly 4 Val Subjects [3, 11, 17, 27]"]
        D --> E["INT8 Static QDQ ONNX<br/>models/optimized/cnn_1d_int8_static.onnx"]
        B --> F["INT8 Dynamic Quantization<br/>models/optimized/cnn_1d_int8.onnx"]
    end

    subgraph Simulation ["Real-Time Edge Streaming Simulation"]
        G["Continuous IMU Stream (50 Hz)<br/>9-Axis Kinematic Telemetry"] --> H["SlidingWindowBuffer<br/>128 Samples (2.56s) / 64 Stride (1.28s)"]
        H --> I["Train-Fitted StandardScaler"]
        I --> J["ONNX Runtime Inference Engine<br/>INT8 Static / FP32"]
        J --> K["Real-Time Activity Inference<br/>Class + Confidence + Latency Check"]
        K --> L{"Processing Budget<br/>Budget: 1280 ms / Actual: ~0.12 ms"}
        L -->|Met (0 missed)| M["Verified Edge-Ready Stream"]
    end
```

### B. Precision & Optimization Benchmark (Official Test Set: 2,947 Windows)

All metrics are empirical results gathered on physical CPU execution with ONNX Runtime 1.32.0 across 1,000 timed samples (5 independent runs of 200 iterations each):

| Model Architecture | Precision Type | Model Size | Size Reduction | Val Acc | Test Acc | Test Macro F1 | Test Weighted F1 | Median Latency | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch Baseline** | FP32 | 410.86 KB | 0.00% | 99.07% | 93.08% | 0.9308 | 0.9300 | 0.6473 ms | 1,404 win/s |
| **ONNX FP32** | FP32 | 420.73 KB | -2.40% | 99.07% | 93.08% | 0.9308 | 0.9300 | 0.1600 ms | 5,609 win/s |
| **ONNX FP16** | FP16 | 222.12 KB | **-45.94%** | 99.07% | 93.08% | 0.9308 | 0.9300 | 0.1443 ms | 6,437 win/s |
| **ONNX INT8 (Static QDQ)** | INT8 Static | 124.94 KB | **-69.59%** | 99.07% | **93.18%** | **0.9318** | **0.9311** | **0.0995 ms** | **9,097 win/s** |
| **ONNX INT8 (Dynamic)** | INT8 Dynamic | 120.98 KB | **-70.56%** | 99.07% | 93.08% | 0.9308 | 0.9300 | 1.1664 ms | 905 win/s |

*Key Empirical & Scientific Insights:*
1. **INT8 Static QDQ achieves the best balance:** 69.59% file size reduction down to 124.94 KB, fastest CPU inference at 0.0995 ms per window (9,097 windows/sec, inter-run std: 0.0048 ms), and 93.18% test accuracy.
2. **Empirical Observation on INT8 Accuracy:** Static INT8 QDQ demonstrated a slight increase in test accuracy from 93.08% to 93.18% (+3 correct windows out of 2,947, primarily in the SITTING and STANDING classes). This difference represents a descriptive observation on this held-out set rather than evidence of superior generalization. Calibration strictly used internal validation subjects (`[3, 11, 17, 27]`), and the test set was completely isolated. The +0.10% delta is within nominal variation for quantization rounding on boundary decision vectors and does not indicate an intrinsic generalization advantage over FP32.
3. **Zero Calibration Snooping:** Calibration used 100 validation windows strictly from the 4 internal validation subjects (`[3, 11, 17, 27]`). Test data was never exposed to calibration.
4. **Dynamic vs Static INT8 on CPU:** Dynamic INT8 is ~10x slower for single-window inference because scale parameters are recalculated dynamically per-window, whereas Static QDQ executes pre-calibrated integer operations.

### C. Real-Time Streaming Emulation Performance

The real-time streaming emulator (`src/deployment/streaming.py`) simulates a real-world wearable device sampling at 50 Hz with 128-sample sliding windows and 50% overlap (64 samples, corresponding to a nominal 1,280 ms stride):

- **Simulated Stream Duration:** 33.28 seconds (1,664 continuous empirical IMU samples).
- **Windows Processed:** 25 windows emitted and classified.
- **Inference Latency:** Mean 0.2309 ms, Median 0.2027 ms, P95 0.3094 ms.
- **Nominal Processing Budget:** 1,280.0 ms.
- **Missed Deadlines:** **0 (0.00%)** — Real-time compute safety margin factor is **> 5,000x**.

### D. Edge Deployment Hardware Feasibility Matrix (Preliminary Analytical Estimates)

Based on detailed memory modeling (see [reports/edge_deployment_feasibility.md](reports/edge_deployment_feasibility.md)):

| Target Platform | Core / Architecture | Clock Speed | SRAM | Flash | Feasibility (Analytical) | Estimated Latency | Working Memory (Est.) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Espressif ESP32-S3** | Dual Xtensa LX7 + SIMD | 240 MHz | 512 KB | 4–16 MB | **Highly Feasible** | 12–25 ms | ~20 KB SRAM (~96% free), ~210 KB Flash |
| **Nordic nRF52840** | ARM Cortex-M4F | 64 MHz | 256 KB | 1 MB | **Highly Feasible** | 45–90 ms | ~20 KB SRAM (~92% free), ~210 KB Flash |
| **Raspberry Pi RP2040** | Dual ARM Cortex-M0+ | 133 MHz | 264 KB | 2 MB | **Feasible (INT8)** | 80–160 ms | ~20 KB SRAM (~92% free), ~210 KB Flash |
| **STM32F401 (BlackPill)** | ARM Cortex-M4F | 84 MHz | 64–96 KB | 256–512 KB | **Feasible** | 35–70 ms | ~20 KB SRAM (leaves 44–76 KB), ~210 KB Flash |

*All microcontroller estimates are analytical projections based on clock cycles and instruction architectures; no physical bare-metal hardware deployment is claimed.*

---

## 11. Visualizations and Artifacts Generated

| Visualization | Description | Figure Path |
| :--- | :--- | :--- |
| **RF Feature Importance** | Top 20 MDI features across 204 engineered dimensions | `reports/figures/rf_feature_importance.png` |
| **CNN Corrected Val Matrix** | Normalized confusion matrix for 1D CNN on leak-free validation split | `reports/figures/cnn_confusion_matrix.png` |
| **CNN Corrected Test Matrix** | Normalized confusion matrix for 1D CNN on 2,947 test windows | `reports/figures/test_cnn_confusion_matrix.png` |
| **RF Corrected Test Matrix** | Normalized confusion matrix for Random Forest on test set | `reports/figures/test_rf_confusion_matrix.png` |
| **SVM Corrected Test Matrix** | Normalized confusion matrix for SVM on test set | `reports/figures/test_svm_confusion_matrix.png` |
| **Accuracy Comparison** | Comparative bar chart of validation accuracy across models | `reports/figures/model_accuracy_comparison.png` |
| **Macro F1 Comparison** | Comparative bar chart of validation Macro F1 across models | `reports/figures/macro_f1_comparison.png` |
| **Training Duration** | Comparison of model training runtimes | `reports/figures/training_time_comparison.png` |
| **Inference Latency** | Per-window inference latency comparison (ms / 2.56s window) | `reports/figures/inference_latency_comparison.png` |
| **Day 3 Static INT8 Test Matrix** | Normalized confusion matrix for ONNX Static INT8 on test set | `reports/day3_confusion_matrices/cm_test_onnx_int8_static_qdq.png` |
| **Day 3 FP16 Test Matrix** | Normalized confusion matrix for ONNX FP16 on test set | `reports/day3_confusion_matrices/cm_test_onnx_fp16.png` |

---

## 12. Automated Testing Results

All 77 automated tests execute cleanly with `pytest` (62% statement coverage across `src/`, ~88% across `src/deployment/`):
```
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.2.0, pluggy-1.7.0
rootdir: .
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.16.0, platformdirs-4.14.8, cov-7.1.0
collected 77 items

tests\test_day3_deployment_and_optimization.py ................          [ 20%]
tests\test_download.py ......                                            [ 28%]
tests\test_end_to_end_pipeline.py ...                                    [ 32%]
tests\test_evaluation.py ..                                              [ 35%]
tests\test_feature_engineering.py .........                              [ 46%]
tests\test_final_training_leakage.py .......                             [ 55%]
tests\test_leakage_audit.py .....                                        [ 62%]
tests\test_loaders.py .....                                              [ 68%]
tests\test_models.py ......                                              [ 76%]
tests\test_preprocessing.py .....                                        [ 83%]
tests\test_simulation.py .....                                           [ 89%]
tests\test_splitting.py ...                                              [ 93%]
tests\test_validation.py .....                                           [100%]

============================= 77 passed in 27.78s =============================
```

See [reports/final_repository_audit.md](reports/final_repository_audit.md) and [reports/day3_final_audit.md](reports/day3_final_audit.md) for comprehensive technical audit reports.

---

## 13. Limitations

1. **Pre-segmented and Filtered Windows:** Signals are pre-filtered with a 0.3 Hz Butterworth filter and segmented into 128-sample windows. They do not represent raw unsegmented continuous ADC data.
2. **Waist Placement Only:** Data was recorded strictly from a waist-mounted smartphone pouch; kinematics differ from wrist-worn wearables.
3. **Controlled Protocol:** Activities were performed under scripted laboratory guidance rather than free-living unconstrained behavior.
4. **No Claim of Physical Hardware Deployment:** Edge profiling and streaming benchmarks are conducted via software emulation on host hardware (x86-64 CPU with ONNX Runtime). Memory footprints and latency projections for microcontrollers are derived from analytical profiling; no physical bare-metal microcontroller flashing or custom PCB fabrication is claimed.
5. **Methodology-Corrected Re-Evaluation Disclosure:** Because official test metrics were observed in Day 2, and subsequently audited in Day 2.1 to remove both preprocessing scaler leakage and CNN validation-monitoring test leakage, the test results represent a rigorous methodology-corrected re-evaluation rather than an untouched blind test.

---

## 14. Project Roadmap

- [x] **Day 1: Data Engineering & Foundation Pipeline** (Automated ingestion, tensor stacking, train-only normalization, exploration report; 29 passing tests at historical milestone).
- [x] **Day 2: Feature Engineering & Model Evaluation** (204 features, subject-wise splitting, Random Forest, SVM, PyTorch 1D CNN, frozen test evaluation; 44 passing tests at historical milestone).
- [x] **Day 2.1: Methodology Audit & Leakage Correction** (Zero-leakage unscaled splitting, internal-only scaler fitting, test re-evaluation; 49 passing tests at historical milestone).
- [x] **Full Repository Repair & Consistency Audit** (Relocation to Drive D, comprehensive suite of 61 passing tests, zero hardcoded paths, exhaustive validation report [reports/final_repository_audit.md](reports/final_repository_audit.md)).
- [x] **Day 3: Model Optimization & Embedded Deployment Simulation** (PyTorch-to-ONNX export, FP16 conversion, INT8 Dynamic & Static QDQ quantization, 77 passing tests, real-time streaming pipeline emulation, edge feasibility audit; see [reports/day3_final_audit.md](reports/day3_final_audit.md) and [reports/edge_deployment_feasibility.md](reports/edge_deployment_feasibility.md)).
