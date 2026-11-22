# Model Validation and Comparative Evaluation Report

**Project:** `imu-human-activity-recognition-ml`  
**Phase:** 7 – Model Evaluation & Comparison  
**Author:** Harish Narayanaan  
**Date:** 2026-11-22  

---

## 1. Executive Summary

This report delivers a rigorous comparative benchmark of three distinct machine learning and deep learning architectures for wearable IMU human activity recognition:
1. **Random Forest Classifier (Ensemble):** 150 estimators trained on 204 engineered statistical & frequency features.
2. **Support Vector Machine (Kernel SVM):** RBF kernel ($C=10.0, \gamma=\text{'scale'}$) with training-fit `StandardScaler`.
3. **1D Deep Convolutional Neural Network (PyTorch):** 3-stage 1D convolution with Batch Normalization, Max Pooling, and Global Average Pooling trained end-to-end directly on raw sensor windows.

All models were evaluated on the **exact same subject-wise validation partition** (4 unseen subjects: `[3, 11, 17, 27]`, 1,401 windows, zero subject overlap with the 17 training subjects).

---

## 2. Quantitative Performance Benchmark

All metrics were genuinely measured from executed code on the validation partition:

| Metric | Random Forest | Support Vector Machine (RBF) | 1D Deep CNN | Top Performer |
| :--- | :--- | :--- | :--- | :--- |
| **Input Representation** | 204 Engineered Features | 204 Scaled Features | Raw 9-Channel Windows | — |
| **Validation Accuracy** | **97.72%** | 96.36% | **98.93%** | **1D CNN** (+1.21% over RF) |
| **Macro F1 Score** | **0.9764** | 0.9620 | **0.9899** | **1D CNN** |
| **Weighted F1 Score** | **0.9771** | 0.9637 | **0.9893** | **1D CNN** |
| **Macro Precision** | 0.9787 | 0.9659 | **0.9910** | **1D CNN** |
| **Macro Recall** | 0.9753 | 0.9607 | **0.9893** | **1D CNN** |
| **Inference Latency** | **0.040 ms / window** | 0.280 ms / window | 0.150 ms / window | **Random Forest** (3.7x faster than CNN) |
| **Training Duration** | 6.24 s | **2.92 s** | 54.24 s | **SVM** |
| **Model Size on Disk** | 6.06 MB | 2.27 MB | **0.40 MB** | **1D CNN** (15x smaller than RF) |
| **Trainable Parameters** | Non-parametric (Trees) | Non-parametric (Dual) | **102,598 parameters** | — |

---

## 3. Per-Class F1 Score Breakdown

| Activity Class | Sample Support | Random Forest F1 | SVM (RBF) F1 | 1D CNN F1 | Primary Confusion Modes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **WALKING** | 235 | 0.9915 | 0.9719 | **0.9979** | Slight confusion with WALKING_UPSTAIRS |
| **WALKING_UPSTAIRS** | 212 | 0.9680 | 0.9401 | **1.0000** | Ascending vs. descending step cadence |
| **WALKING_DOWNSTAIRS** | 185 | 0.9665 | 0.9459 | **0.9973** | Impact dynamics overlap with upstairs |
| **SITTING** | 239 | 0.9633 | 0.9526 | **0.9698** | Posture tilt overlap with STANDING |
| **STANDING** | 266 | 0.9689 | 0.9615 | **0.9744** | Micro-movements during quiet standing |
| **LAYING** | 264 | **1.0000** | **1.0000** | **1.0000** | Perfect separation across all models |

---

## 4. Architectural Analysis and Trade-Offs

### A. 1D Convolutional Neural Network (PyTorch)
- **Strengths:** 
  - Achieved the highest validation accuracy (98.93%) and macro F1 (0.9899).
  - Learns hierarchical temporal receptive fields directly from raw signal waveforms without requiring feature engineering or manual signal processing.
  - Exceptionally compact on-disk footprint (**0.40 MB**), making it highly suitable for constrained edge deployment (microcontrollers, mobile operating systems).
- **Trade-offs:** Longest training time (54.24s) and requires PyTorch/C++ runtime support.

### B. Random Forest Classifier
- **Strengths:**
  - Lightning-fast inference latency (**0.040 ms per 2.56-second window**), running at over 25,000 predictions per second on a single CPU core.
  - Highly robust to non-linear feature interactions and provides intrinsic feature importance interpretability.
- **Trade-offs:** Model file size (6.06 MB) scales with the number of trees and maximum depth.

### C. Support Vector Machine (RBF Kernel)
- **Strengths:** 
  - Rapid convergence and training time (2.92s).
  - Clean mathematical margin maximization.
- **Trade-offs:** Scales quadratically with sample size $O(N^2)$; slightly more sensitive to borderline static posture distinctions (SITTING vs. STANDING).

---

## 5. Artifacts and Visualization References

The complete benchmarking figures are available in `reports/figures/`:
- `reports/figures/rf_confusion_matrix.png`
- `reports/figures/svm_confusion_matrix.png`
- `reports/figures/cnn_confusion_matrix.png`
- `reports/figures/model_accuracy_comparison.png`
- `reports/figures/macro_f1_comparison.png`
- `reports/figures/training_time_comparison.png`
- `reports/figures/inference_latency_comparison.png`
- `reports/figures/rf_feature_importance.png`
- `reports/model_comparison.csv`
