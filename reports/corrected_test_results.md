# Corrected Official Test-Set Results and Re-Evaluation

**Project:** `imu-human-activity-recognition-ml`
**Phase:** Day 2.1 – Methodology Correction & Independent Test-Set Audit
**Evaluation Protocol:** Final models retrained on full official training partition (21 subjects, 7,352 windows) with `ChannelWiseStandardScaler` fitted exclusively on official training data. Evaluated on official frozen test partition (9 unseen subjects, 2,947 windows).
**Official Test Subjects:** `[2, 4, 9, 10, 12, 13, 18, 20, 24]`
**Author:** Harish Narayanaan
**Date:** 2026-11-22

---

## 1. Important Scientific Qualification & Audit Disclosure

The official test metrics were observed during the Day 2 initial publication. Furthermore, an independent code audit identified that in the initial Day 2.1 code, the final CNN training loop inadvertently passed test arrays into `fit()` as validation monitoring data, enabling test-loss checkpoint selection.

**Correction Implemented:**
- In this revised evaluation, `CNN1DHAR.fit_fixed_epochs()` was implemented and enforced.
- The final CNN training schedule was strictly fixed at **32 epochs**, derived exclusively from the internal validation experiment (`cnn_val.best_epoch_ = 32`).
- **Zero test arrays or labels were passed to the training routine.** No test loss was computed during training, and zero checkpoint selection occurred based on test feedback.
- The official test partition was loaded and passed exclusively to `predict()` once training was completed.
- Consequently, this evaluation is designated as a **methodology-corrected re-evaluation** rather than a blind first-time evaluation.

---

## 2. Quantitative Final Test Results

| Model Architecture | Input Modality | Corrected Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Test Latency | Model Size | Trainable Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D Deep CNN** | Raw 9-Ch Waveforms | **93.08%** | **0.9304** | **0.9322** | **0.9308** | **0.9300** | 0.157 ms / win | **0.40 MB** | 102,598 |
| **SVM (RBF Kernel)** | 204 Scaled Feats | **92.20%** | 0.9230 | 0.9215 | 0.9217 | 0.9218 | 0.341 ms / win | 2.46 MB | Non-parametric |
| **Random Forest** | 204 Engineered Feats | **91.38%** | 0.9148 | 0.9113 | 0.9124 | 0.9136 | **0.025 ms / win** | 6.92 MB | Non-parametric |

---

## 3. Comparison with Historical Test Results

| Model Architecture | Historical Test Accuracy (Leaked) | Premature Test Checkpoint | Corrected Test Accuracy (Strict Fixed-Epoch) | Final Impact Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **1D Deep CNN** | 93.15% | 93.38% (Test loss min @ ep 2) | **93.08%** (Strict 32 epochs, zero test leakage) | Genuine out-of-sample performance without post-hoc checkpoint cherry-picking. |
| **SVM (RBF Kernel)** | 92.16% | 92.20% | **92.20%** | Robust deterministic margin classifier across full 7,352 samples. |
| **Random Forest** | 91.62% | 91.38% | **91.38%** | Slight stochastic variance across 150 bagged trees. |

---

## 4. Per-Class Performance Breakdown (Final Test Partition)

### A. 1D Deep CNN (PyTorch)
```
                    precision    recall  f1-score   support

           WALKING     0.9939    0.9940    0.9939       496
  WALKING_UPSTAIRS     0.9430    0.9490    0.9460       471
WALKING_DOWNSTAIRS     0.9354    1.0000    0.9666       420
           SITTING     0.8549    0.7800    0.8158       491
          STANDING     0.8488    0.8759    0.8622       532
            LAYING     1.0000    1.0000    1.0000       537

          accuracy                         0.9308      2947
         macro avg     0.9304    0.9322    0.9308      2947
      weighted avg     0.9302    0.9308    0.9300      2947
```

### B. Support Vector Machine (RBF Kernel)
```
                    precision    recall  f1-score   support

           WALKING     0.8857    0.9214    0.9032       496
  WALKING_UPSTAIRS     0.9524    0.8917    0.9211       471
WALKING_DOWNSTAIRS     0.9349    0.9571    0.9459       420
           SITTING     0.9059    0.8432    0.8734       491
          STANDING     0.8665    0.9154    0.8903       532
            LAYING     0.9926    1.0000    0.9963       537

          accuracy                         0.9220      2947
         macro avg     0.9230    0.9215    0.9217      2947
      weighted avg     0.9227    0.9220    0.9218      2947
```

### C. Random Forest Classifier
```
                    precision    recall  f1-score   support

           WALKING     0.8796    0.9577    0.9170       496
  WALKING_UPSTAIRS     0.9208    0.9130    0.9168       471
WALKING_DOWNSTAIRS     0.9424    0.8571    0.8978       420
           SITTING     0.8779    0.8493    0.8634       491
          STANDING     0.8681    0.8910    0.8794       532
            LAYING     1.0000    1.0000    1.0000       537

          accuracy                         0.9138      2947
         macro avg     0.9148    0.9113    0.9124      2947
      weighted avg     0.9147    0.9138    0.9136      2947
```

---

## 5. Artifact Verification
- Final models saved to:
  - `models/random_forest.joblib` (6.92 MB)
  - `models/svm.joblib` (2.46 MB)
  - `models/cnn_1d.pt` (0.40 MB)
- Test confusion matrix plots generated at:
  - `reports/figures/test_rf_confusion_matrix.png`
  - `reports/figures/test_svm_confusion_matrix.png`
  - `reports/figures/test_cnn_confusion_matrix.png`
