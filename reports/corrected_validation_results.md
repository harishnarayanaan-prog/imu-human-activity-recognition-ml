# Corrected Subject-Wise Validation Results (Zero Data Leakage)

**Project:** `imu-human-activity-recognition-ml`  
**Phase:** Day 2.1 – Methodology Correction  
**Evaluation Protocol:** Subject-wise split on unscaled raw sensor windows. `ChannelWiseStandardScaler` fitted strictly on internal training partition. Zero participant overlap.  
**Validation Subjects:** 4 unseen subjects (`[3, 11, 17, 27]`, 1,401 windows).  
**Internal Training Subjects:** 17 subjects (`[1, 5, 6, 7, 8, 14, 15, 16, 19, 21, 22, 23, 25, 26, 28, 29, 30]`, 5,951 windows).  
**Author:** Harish Narayanaan  
**Date:** 2026-11-22  

---

## 1. Quantitative Benchmark Summary

All metrics were genuinely measured from physical CPU execution using the leak-free validation pipeline:

| Model Architecture | Input Features | Validation Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Inference Latency | Model Size | Training Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D Deep CNN** | Raw 9-Ch Waveforms | **99.07%** | **0.9922** | **0.9909** | **0.9914** | **0.9907** | 0.163 ms / win | **0.40 MB** | 138.64 s |
| **Random Forest** | 204 Engineered Feats | **97.86%** | 0.9801 | 0.9761 | 0.9775 | 0.9785 | **0.017 ms / win** | 6.06 MB | 6.38 s |
| **SVM (RBF Kernel)** | 204 Scaled Feats | **96.29%** | 0.9653 | 0.9597 | 0.9611 | 0.9630 | 0.225 ms / win | 2.27 MB | **2.98 s** |

---

## 2. Comparison with Historical Benchmark (Leakage vs. Leak-Free)

| Model Architecture | Historical Val Accuracy (Leaked) | Corrected Val Accuracy (Leak-Free) | Delta ($\Delta$) | Impact Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **1D Deep CNN** | 98.93% | **99.07%** | +0.14% | Deep spatio-temporal features are highly robust to independent scaling. |
| **Random Forest** | 97.72% | **97.86%** | +0.14% | Decision trees split on rank order; minor scaling shift has negligible impact. |
| **SVM (RBF Kernel)** | 96.36% | **96.29%** | -0.07% | Minor expected reduction due to strict isolation of validation margin support. |

---

## 3. Per-Class F1 Score Breakdown (Corrected Validation)

| Activity Class | Sample Support | Random Forest F1 | SVM (RBF) F1 | 1D CNN F1 |
| :--- | :--- | :--- | :--- | :--- |
| **WALKING** | 235 | 0.9915 | 0.9719 | **1.0000** |
| **WALKING_UPSTAIRS** | 212 | 0.9680 | 0.9401 | **0.9835** |
| **WALKING_DOWNSTAIRS** | 185 | 0.9701 | 0.9459 | **0.9973** |
| **SITTING** | 239 | 0.9633 | 0.9526 | **0.9749** |
| **STANDING** | 266 | 0.9723 | 0.9598 | **0.9925** |
| **LAYING** | 264 | **1.0000** | **1.0000** | **1.0000** |

---

## 4. Leak-Free Validation Guarantees

1. `ChannelWiseStandardScaler` parameters ($\mu_c, \sigma_c$) were computed **strictly on the 5,951 windows of the 17 internal training subjects**.
2. Validation windows (1,401 windows from subjects 3, 11, 17, 27) were standardized using the frozen training statistics.
3. Feature extraction (204 dimensions) was computed locally within each window of the respective transformed partitions.
4. For SVM, `StandardScaler` inside the scikit-learn `Pipeline` was fitted strictly on the internal training feature matrix.
