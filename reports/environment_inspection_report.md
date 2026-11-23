# Machine Learning Environment Inspection & Configuration Report

**Project:** Human Activity Recognition from Wearable IMU Sensor Data
**Location:** `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml`
**Execution Date:** 2026-11-23
**Auditor:** Senior ML & MLOps Specialist

---

## 1. System & Hardware Specifications

| Component | Specification / Value |
| :--- | :--- |
| **Operating System** | Windows 11 Enterprise (Build 10.0.26300, 64-bit AMD64) |
| **Host Storage (Drive D:)** | Total: **781.5 GB** \| Free: **465.0 GB** (Sufficient for datasets, tensors, model checkpoints) |
| **Git Version** | `git version 2.56.1.windows.1` |
| **GPU / Acceleration** | CUDA Available: **False** (CPU Execution: Intel/AMD Multi-core Architecture) |
| **PyTorch Execution Target** | **CPU** (`torch 2.15.0+cpu`) |

---

## 2. Python Virtual Environment Configuration

- **Environment Path:** `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml\.venv`
- **Python Binary:** `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml\.venv\Scripts\python.exe`
- **Python Version:** `3.14.3` (`MSC v.1944 64 bit (AMD64)`)
- **Package Manager:** `pip 26.3`

---

## 3. Installed Package Inventory & Compatibility Matrix

All packages verified through direct runtime module import (`__import__`):

| Package Name | Installed Version | Role in Pipeline | Compatibility Status |
| :--- | :--- | :--- | :--- |
| `numpy` | `2.5.5` | 3D array tensor manipulation `(samples, timesteps, channels)` | Verified Compatible |
| `pandas` | `3.1.1` | Tabular feature indexing, CSV metrics serialization | Verified Compatible |
| `scipy` | `1.18.1` | Butterworth digital filtering, FFT spectral analysis, skew/kurtosis | Verified Compatible |
| `scikit-learn` | `1.9.1` | Random Forest, RBF SVM, classification metrics, confusion matrices | Verified Compatible |
| `matplotlib` | `3.11.2` | Visualization generation (confusion matrices, bar charts) | Verified Compatible |
| `seaborn` | `0.13.2` | Heatmap confusion matrix rendering | Verified Compatible |
| `joblib` | `1.6.0` | Serialization of scikit-learn model artifacts (`.joblib`) | Verified Compatible |
| `torch` | `2.15.0+cpu` | 1D Deep Convolutional Neural Network (PyTorch engine) | Verified Compatible (CPU) |
| `pytest` | `9.2.0` | Automated unit, regression, and pipeline test runner | Verified Compatible |
| `pytest-cov` | `7.1.0` | Test suite code coverage profiling | Verified Compatible |
| `ipykernel` | `7.4.0` | Jupyter interactive notebook kernel execution | Verified Compatible |
| `jupyter` | `1.1.1` | Interactive notebook interface | Verified Compatible |
| `tqdm` | `4.70.1` | Progress visualization for feature extraction & tensor loading | Verified Compatible |
| `requests` | `2.34.2` | Automated dataset retrieval and SHA-256 verification | Verified Compatible |
| `onnx` | `1.23.2` | Open Neural Network Exchange graph specification (Day 3 ready) | Verified Compatible |
| `onnxruntime` | `1.32.0` | High-performance inference engine for ONNX runtimes | Verified Compatible |
| `onnxscript` | `0.7.2` | PyTorch to ONNX graph translation utilities | Verified Compatible |

---

## 4. Environment Verification & Reproduction Protocol

To reconstruct the identical virtual environment on a clean system:
```powershell
# Navigate to project location on Drive D
cd D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml

# Create virtual environment using Python 3.14
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install complete development and deployment dependencies
pip install -r requirements-dev.txt
```
