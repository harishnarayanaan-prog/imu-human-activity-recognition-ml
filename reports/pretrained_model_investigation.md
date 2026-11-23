# Pretrained Model Baseline Feasibility & Provenance Investigation

**Project:** Human Activity Recognition from Wearable IMU Sensor Data
**Location:** `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml`
**Date:** November 23, 2026
**Auditor:** Senior ML Engineer & Independent Reviewer

---

## 1. Objective

Phase 9 evaluates whether a publicly downloadable pretrained Human Activity Recognition (HAR) checkpoint should be incorporated as an external baseline alongside the project's native three-model architecture (Random Forest, SVM, 1D Deep CNN).

Per the project's strict audit protocol, an external pretrained model may **only** be integrated if it satisfies all 7 mandatory scientific criteria:
1. **Identified Dataset Provenance:** Originating dataset is verifiable and documented.
2. **Documented Architecture:** Exact layer dimensions, activations, and parameter count are published.
3. **Compatible Preprocessing:** Channel order, sampling rate (50 Hz), window duration (128 samples / 2.56 s), and normalization ($\mu, \sigma$) match the UCI HAR format.
4. **Documented Split Protocol:** Verifiable subject-wise isolation; no overlap between the pretrained training set and our official test set.
5. **Permissive License:** License explicitly permits redistribution and research benchmarking.
6. **Cryptographically Verifiable Source:** Checkpoint weights possess official SHA-256 hashes from an authoritative source.
7. **Reproducible Performance:** Empirical claims can be replicated locally without data snooping.

---

## 2. Evaluation of Available Public Checkpoints

We investigated candidates from PyTorch Hub, Hugging Face Hub, and open-source GitHub repositories:

### Candidate A: Community Hugging Face UCI-HAR Checkpoints
- **Findings:** Multiple community models (e.g. MLP / 1D-CNN checkpoints) claim $>96\%$ accuracy on UCI-HAR.
- **Flaws Identified:**
  1. *Input Incompatibility:* Many models expect the 561-dimensional manual feature vector from `X_train.txt` rather than raw 9-channel time-series tensors `(samples, 128, 9)`.
  2. *Severe Data Leakage:* Most models were trained using random `train_test_split(test_size=0.3, shuffle=True)` across overlapping 50% sliding windows. Adjacent 128-sample windows share 64 identical physical samples, causing profound identity and temporal leakage that artificially inflates test accuracy.
  3. *Unverifiable Split Boundaries:* Participant IDs were ignored during training, meaning test volunteers contributed training samples.
  4. *Lack of Weight Signatures:* Checkpoint weights lack cryptographic SHA-256 checksums from official research institutions.

### Candidate B: Generic Time-Series Foundation Models (e.g., Chronos, TimesNet, MOMENT)
- **Findings:** Zero-shot and pre-trained time-series foundation models exist for forecasting and anomaly detection.
- **Flaws Identified:**
  1. Primarily pre-trained for single-channel or multi-channel forecasting rather than 6-class wearable inertial kinematics classification.
  2. Model sizes exceed 50 MB to 1 GB, violating edge microcontroller feasibility constraints (target: $< 1\text{ MB}$, $< 150\text{ KB}$ RAM for Day 3 edge optimization).

---

## 3. Decision & Technical Justification

> [!IMPORTANT]
> **Verdict: Omit External Pretrained Checkpoints.**
> No external pretrained checkpoint satisfies all 7 scientific criteria without introducing data leakage, input incompatibility, or provenance ambiguity.
>
> Downloading an opaque, unverifiable pretrained checkpoint would undermine the rigorous, zero-leakage methodology established in Day 1, Day 2, and Day 2.1.

### Preservation of the Project's Native Three-Model Zoo
The project's internal 3-model benchmark provides an uncompromised, reproducible, self-contained comparative evaluation:
1. **Random Forest Classifier:** 204 engineered time/frequency features | **91.38%** test accuracy | 0.032 ms inference latency.
2. **Support Vector Machine (RBF):** 204 standardized features | **92.20%** test accuracy | 0.353 ms inference latency.
3. **1D Deep CNN (PyTorch):** Raw 9-channel waveform tensors | **93.08%** test accuracy | 0.229 ms inference latency | 102,598 parameters (0.40 MB).

All three models are trained completely from scratch with deterministic seeds, zero validation snooping, and zero official test leakage.
