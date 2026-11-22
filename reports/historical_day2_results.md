# Historical Day 2 Benchmark Results (Pre-Correction Baseline)

**Project:** `imu-human-activity-recognition-ml`  
**Archived Baseline:** Day 2 Initial Publication (2026-11-22)  
**Status:** Archived Historical Benchmark (Superseded by Day 2.1 Corrected Methodology)  

---

## 1. Notice of Methodological Flaw in Historical Benchmark

In the initial Day 2 release, `ChannelWiseStandardScaler` was fitted across the entire official training partition (21 subjects, 7,352 windows) *prior* to partitioning into internal training (17 subjects, 5,951 windows) and validation (4 subjects: `[3, 11, 17, 27]`, 1,401 windows). 

Consequently, the 4 validation subjects contributed to the global channel standardization parameters ($\mu_c, \sigma_c$), constituting subtle data leakage prior to validation evaluation. 

The historical metrics recorded under that protocol are archived below for transparency and comparative audit.

---

## 2. Historical Validation Results (Subject to Preprocessing Leakage)

*Evaluated on 4 validation subjects (`[3, 11, 17, 27]`, 1,401 windows) from globally scaled training tensors:*

| Model Architecture | Input Representation | Historical Accuracy | Historical Macro F1 | Historical Latency | Model Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1D Deep CNN** | Globally Scaled Raw Windows | 98.93% | 0.9899 | 0.150 ms / win | 0.40 MB |
| **Random Forest** | 204 Features (from Global Scale) | 97.72% | 0.9764 | 0.040 ms / win | 6.06 MB |
| **SVM (RBF Kernel)** | 204 Scaled Feats (Double Scale) | 96.36% | 0.9620 | 0.280 ms / win | 2.27 MB |

---

## 3. Historical Held-Out Test Results (Observed Prior to Correction)

*Evaluated on 9 held-out test subjects (2,947 windows):*

| Model Architecture | Input Modality | Historical Test Accuracy | Historical Macro F1 | Historical Latency |
| :--- | :--- | :--- | :--- | :--- |
| **1D Deep CNN** | Official Scaled Raw Windows | 93.15% | 0.9326 | 0.173 ms / win |
| **SVM (RBF Kernel)** | 204 Scaled Features | 92.16% | 0.9212 | 0.266 ms / win |
| **Random Forest** | 204 Features | 91.62% | 0.9149 | 0.035 ms / win |

---

## 4. Preservation Note

These historical figures represent the state of the repository at commit `011b4b0ce77ed4f36b52fdd7a38e4eecf38df232`. For the mathematically clean, leak-free pipeline and its genuine empirical results, consult:
- `reports/methodology_correction_report.md`
- `reports/corrected_validation_results.md`
- `reports/corrected_test_results.md`
