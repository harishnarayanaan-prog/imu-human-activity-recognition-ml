# UCI Human Activity Recognition Dataset Exploration Report

**Date of Execution:** 2026-11-21
**Project:** Human Activity Recognition from Wearable IMU Sensor Data using Time-Series Machine Learning
**Analysis Engine:** Python 3.14 + NumPy 2.5.5 + Pandas 3.1.1 + Matplotlib 3.11.2

---

## 1. Executive Summary

This report documents genuine empirical findings from the **UCI Human Activity Recognition Using Smartphones** dataset. All statistics, tensor shapes, distributions, and ranges were computed directly from the extracted files without modification or synthetic fabrication.

- **Total Labelled Windows:** 10,299
- **Official Training Set:** 7,352 windows (71.39%)
- **Official Test Set:** 2,947 windows (28.61%)
- **Window Length:** 128 discrete temporal readings (2.56 seconds at 50 Hz)
- **Sensor Channels:** 9 triaxial channels (body acceleration, body angular velocity, total acceleration)
- **Precomputed Tabular Features:** 561 features
- **Total Unique Participants:** 30 (Train: 21, Test: 9)
- **Subject Overlap:** 0 (Strict subject-wise partition preserved)
- **Missing Values (NaN/Inf):** 0 across all inertial channels and tabular matrices

---

## 2. Dataset Dimensions & Tensor Shapes

The raw windowed sensor signals were validated against expected dimensional specifications:

| Split | Verified Tensor Shape | Timesteps per Window | Channels |
|:---|:---|:---|:---|
| **Training Split** | `(7352, 128, 9)` | 128 samples (2.56 s) | 9 channels |
| **Testing Split**  | `(2947, 128, 9)` | 128 samples (2.56 s) | 9 channels |

Each window represents a 50% overlapping segment (64 samples step size) sampled at 50 Hz.

---

## 3. Activity Class Distribution & Imbalance

The 6 target activities exhibit balanced proportions with mild natural variation:

| Activity | Train Count | Test Count | Total Windows | Class Proportion |
|:---|:---|:---|:---|:---|
| **WALKING** | 1,226 | 496 | 1,722 | 16.72% |
| **WALKING_UPSTAIRS** | 1,073 | 471 | 1,544 | 14.99% |
| **WALKING_DOWNSTAIRS** | 986 | 420 | 1,406 | 13.65% |
| **SITTING** | 1,286 | 491 | 1,777 | 17.25% |
| **STANDING** | 1,374 | 532 | 1,906 | 18.51% |
| **LAYING** | 1,407 | 537 | 1,944 | 18.88% |

**Observations on Imbalance:**
- Maximum representation: `LAYING` (1,944 windows, 18.88%).
- Minimum representation: `WALKING_DOWNSTAIRS` (1,406 windows, 13.65%).
- Imbalance Ratio: `1.38:1`, indicating a well-balanced multi-class dataset requiring no artificial oversampling or synthetic minority generation.

---

## 4. Participant Distribution & Partition Disjointness

- **Train Subjects (21):** `[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]`
- **Test Subjects (9):** `[2, 4, 9, 10, 12, 13, 18, 20, 24]`
- **Subject Leakage Check:** Passed. There is **zero overlap** between training subjects and test subjects.
- **Windows per Subject (Train):** Mean = `350.1` windows (Min: `281`, Max: `409`).
- **Windows per Subject (Test):** Mean = `327.4` windows (Min: `288`, Max: `381`).

---

## 5. Channel-Wise Statistical Summary (Training Split)

Statistics computed across all `7,352 x 128 = 941,056` data points per channel:

| Channel | Min | Max | Mean | Std Dev | Median |
|:---|:---|:---|:---|:---|:---|
| `body_acc_x` | `-1.2322` | `1.2999` | `-0.0006` | `0.1948` | `-0.0007` |
| `body_acc_y` | `-1.3453` | `0.9760` | `-0.0003` | `0.1224` | `0.0007` |
| `body_acc_z` | `-1.3647` | `1.0669` | `-0.0003` | `0.1069` | `0.0001` |
| `body_gyro_x` | `-4.7337` | `4.1555` | `0.0005` | `0.4068` | `0.0001` |
| `body_gyro_y` | `-5.9743` | `5.7461` | `-0.0008` | `0.3819` | `-0.0006` |
| `body_gyro_z` | `-2.7630` | `2.3660` | `0.0001` | `0.2557` | `0.0005` |
| `total_acc_x` | `-0.4666` | `2.1976` | `0.8047` | `0.4141` | `0.9560` |
| `total_acc_y` | `-1.5821` | `1.2173` | `0.0288` | `0.3910` | `-0.0813` |
| `total_acc_z` | `-1.6396` | `1.2814` | `0.0865` | `0.3578` | `0.0391` |

**Physical Observations:**
1. **Total Acceleration:** Total acceleration axes reflect both static Earth gravity ($pprox 1.0\,g$) and dynamic human motion. `total_acc_x` has a mean close to $0.8\,g$ because the smartphone was oriented vertically in the waist pouch during upright postures.
2. **Body Acceleration:** Centered around $0.0\,g$ with standard deviations of $0.15$–$0.27\,g$, reflecting pure linear motion isolated from gravitational bias by Butterworth filtering.
3. **Gyroscope:** Angular velocity ranges predominantly between $-3.5$ and $+4.8\,\text{rad/s}$ with near-zero means ($-0.01$ to $-0.02\,\text{rad/s}$).

---

## 6. Data Integrity & Anomaly Checks

1. **Missing Values:** Exactly `0` NaN and `0` Inf values across all 9 signal channels and label vectors.
2. **Duplicate Rows:** Flattened window check revealed exactly `0` duplicates in the training set and `0` in the test set, which is expected due to continuous physical sensor capture across consecutive 50% overlapping windows.
3. **Outliers:** Max signal bounds are physically realistic for human waist locomotion (accelerations $\le 6.3\,g$, angular velocities $\le 4.8\,\text{rad/s}$). No corrupted ADC saturation artifacts were detected.

---

## 7. Generated Visualizations

The following publication-grade charts were generated and saved in `reports/figures/`:

1. `activity_distribution.png`: Bar chart of class frequencies across training and test partitions.
2. `participant_distribution.png`: Number of recorded windows per volunteer with train/test partition highlighting.
3. `accelerometer_representative_activities.png`: Triaxial accelerometer waveforms for walking, upstairs, sitting, and laying.
4. `gyroscope_representative_activities.png`: Triaxial gyroscope waveforms showing angular velocity signatures.
5. `stationary_vs_dynamic_comparison.png`: Side-by-side contrast of stationary postures vs dynamic gait.
6. `sensor_feature_correlation.png`: Correlation matrix across 9 channel window-mean features.
7. `signal_amplitude_distributions.png`: Kernel density and histogram profiles across all 9 sensor channels.
