# Feature Engineering and Extraction Report

**Project:** `imu-human-activity-recognition-ml`  
**Phase:** 2 – Feature Engineering  
**Author:** Harish Narayanaan  
**Date:** 2026-11-22  

---

## 1. Executive Summary

This report documents the feature engineering pipeline developed for the triaxial accelerometer and gyroscope time-series signals from the UCI Human Activity Recognition (HAR) dataset. 

A high-performance vectorized feature extraction engine (`src/features/feature_engineering.py`) was engineered to extract 17 distinct statistical and frequency-domain representations from each of 12 IMU channels (9 raw channels + 3 derived Euclidean magnitudes), yielding a total of **204 engineered features** per 128-sample (2.56-second) window.

The extraction pipeline processed all 7,352 training windows and 2,947 testing windows with 100% data integrity (zero NaNs, zero Infs) in under 3 seconds.

---

## 2. Input Channels and Derived Representations

### A. Base Triaxial Inertial Channels (9 Channels)
Captured by smartphone sensors at a constant sampling rate of 50 Hz:
1. `body_acc_x`: Estimated body linear acceleration along the X-axis ($g$).
2. `body_acc_y`: Estimated body linear acceleration along the Y-axis ($g$).
3. `body_acc_z`: Estimated body linear acceleration along the Z-axis ($g$).
4. `body_gyro_x`: Angular velocity measured by gyroscope along the X-axis ($\text{rad/s}$).
5. `body_gyro_y`: Angular velocity measured by gyroscope along the Y-axis ($\text{rad/s}$).
6. `body_gyro_z`: Angular velocity measured by gyroscope along the Z-axis ($\text{rad/s}$).
7. `total_acc_x`: Total acceleration along the X-axis ($g$).
8. `total_acc_y`: Total acceleration along the Y-axis ($g$).
9. `total_acc_z`: Total acceleration along the Z-axis ($g$).

### B. Derived Euclidean Magnitude Channels (3 Channels)
To achieve rotational invariance and capture total motion intensity regardless of sensor coordinate orientation:
10. `body_acc_mag`: $\sqrt{\text{body\_acc\_x}^2 + \text{body\_acc\_y}^2 + \text{body\_acc\_z}^2}$
11. `body_gyro_mag`: $\sqrt{\text{body\_gyro\_x}^2 + \text{body\_gyro\_y}^2 + \text{body\_gyro\_z}^2}$
12. `total_acc_mag`: $\sqrt{\text{total\_acc\_x}^2 + \text{total\_acc\_y}^2 + \text{total\_acc\_z}^2}$

---

## 3. Mathematical Formulations of Extracted Features

For each of the 12 channels $x = [x_1, x_2, \dots, x_N]$ where $N = 128$:

### A. Statistical Time-Domain Features (13)

| Feature | Mathematical Definition | Physical Significance |
| :--- | :--- | :--- |
| **Mean** | $\mu = \frac{1}{N}\sum_{i=1}^N x_i$ | DC offset; static posture orientation with respect to gravity. |
| **Standard Deviation** | $\sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i - \mu)^2}$ | Dynamic motion intensity and movement variation. |
| **Variance** | $\sigma^2 = \frac{1}{N-1}\sum_{i=1}^N (x_i - \mu)^2$ | Dispersion of sensor readings around the mean. |
| **Minimum** | $\min(x)$ | Lower boundary of dynamic envelope. |
| **Maximum** | $\max(x)$ | Peak impact or extreme acceleration spike. |
| **Median** | $\text{median}(x)$ | Robust central tendency resistant to transient anomalies. |
| **Range (PTP)** | $\max(x) - \min(x)$ | Full dynamic excursion within the time window. |
| **Root Mean Square (RMS)** | $\sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}$ | Quadratic mean; overall power and signal strength. |
| **Mean Absolute Value (MAV)** | $\frac{1}{N}\sum_{i=1}^N \|x_i\|$ | Rectified signal magnitude, common in biomechanical kinematics. |
| **Interquartile Range (IQR)** | $Q_{75} - Q_{25}$ | Middle 50% spread, robust non-parametric variance proxy. |
| **Signal Energy** | $\frac{1}{N}\sum_{i=1}^N x_i^2$ | Mean kinetic energy content of the movement. |
| **Skewness** | $\frac{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^3}{\sigma^3}$ | Asymmetry of probability density function (gait asymmetry). |
| **Kurtosis** | $\frac{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^4}{\sigma^4} - 3$ | "Peakedness" and tail heaviness; discriminates impulsive impacts from smooth cyclic actions. |

*Zero-variance protection:* When $\sigma < 10^{-8}$, skewness and kurtosis are analytically clipped to 0.0 to prevent division by zero.

### B. Frequency-Domain Features (4)

Applying the Real Fast Fourier Transform ($\text{RFFT}$):
$$X_k = \sum_{n=0}^{N-1} x_n e^{-j 2\pi k n / N}, \quad k = 0, 1, \dots, \frac{N}{2}$$
Frequencies correspond to $f_k = k \cdot \frac{f_s}{N}$ with $f_s = 50.0\text{ Hz}$ across 65 discrete bins ($0\text{ Hz}$ to $25\text{ Hz}$).
Power Spectral Density: $P_k = \|X_k\|^2$.

| Feature | Mathematical Definition | Physical Significance |
| :--- | :--- | :--- |
| **Dominant Frequency** | $f_{\text{dom}} = f_{\arg\max_k P_k}$ | Cadence of cyclic movements (e.g. 1.5–2.5 Hz in human walking). |
| **Spectral Energy** | $E_{\text{spec}} = \frac{1}{N}\sum_{k} P_k$ | Total harmonic energy across the frequency spectrum. |
| **Spectral Entropy** | $H_{\text{spec}} = -\sum_{k} p_k \log_2(p_k)$, where $p_k = \frac{P_k}{\sum_j P_j + \epsilon}$ | Measure of spectral complexity (peaked for periodic walking, flat for noise/transitions). |
| **Frequency Centroid** | $f_{\text{cent}} = \frac{\sum_k f_k P_k}{\sum_k P_k + \epsilon}$ | Spectral center of mass; captures shift toward high-frequency tremor or low-frequency posture shifts. |

---

## 4. Feature Extraction Output Characteristics

| Split | Input Tensor Shape | Extracted Feature Matrix | Extracted Size on Disk | NaNs / Infs |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set** | `(7352, 128, 9)` | `(7352, 204)` | 11.99 MB | 0 / 0 |
| **Testing Set** | `(2947, 128, 9)` | `(2947, 204)` | 4.81 MB | 0 / 0 |

The engineered feature matrices and complete metadata schema are saved at:
- `data/processed/X_train_features.npy`
- `data/processed/X_test_features.npy`
- `data/processed/feature_metadata.json`
