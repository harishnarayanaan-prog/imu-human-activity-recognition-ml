# Data Dictionary & Schema Specification

## 1. Sensor Channels (9 Canonical Channels)

Each window consists of 128 discrete temporal measurements sampled at 50 Hz (20 ms interval) over a 2.56-second sliding window with 50% overlap (64 samples).

| Channel Index | Channel Identifier | Sensor Modality | Units | Physical Interpretation | Signal Range (Train) |
|:---:|:---|:---|:---:|:---|:---:|
| **0** | `body_acc_x` | Accelerometer | $g$ ($9.80665\,\text{m/s}^2$) | Triaxial body linear acceleration along lateral axis (isolated via Butterworth filter) | $[-1.4082, 1.4883]$ |
| **1** | `body_acc_y` | Accelerometer | $g$ | Triaxial body linear acceleration along vertical axis | $[-1.7058, 2.0520]$ |
| **2** | `body_acc_z` | Accelerometer | $g$ | Triaxial body linear acceleration along forward axis | $[-1.7583, 1.4841]$ |
| **3** | `body_gyro_x` | Gyroscope | $\text{rad/s}$ | Angular velocity around X-axis (pitch/roll dynamics) | $[-4.8055, 3.8643]$ |
| **4** | `body_gyro_y` | Gyroscope | $\text{rad/s}$ | Angular velocity around Y-axis | $[-3.0538, 4.3061]$ |
| **5** | `body_gyro_z` | Gyroscope | $\text{rad/s}$ | Angular velocity around Z-axis (yaw/torsion) | $[-3.1557, 2.7686]$ |
| **6** | `total_acc_x` | Accelerometer | $g$ | Device acceleration including Earth gravitational vector | $[-0.4682, 6.2885]$ |
| **7** | `total_acc_y` | Accelerometer | $g$ | Device acceleration including Earth gravitational vector | $[-1.9360, 1.6377]$ |
| **8** | `total_acc_z` | Accelerometer | $g$ | Device acceleration including Earth gravitational vector | $[-1.7378, 1.9388]$ |

---

## 2. Activity Label Encodings

| Raw ID (UCI) | Model Target (0-Indexed) | Activity Name | Locomotion Category | Dynamic / Static | Total Windows | Train Split | Test Split |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|
| `1` | `0` | `WALKING` | Horizontal Gait | Dynamic | 1,722 | 1,226 | 496 |
| `2` | `1` | `WALKING_UPSTAIRS` | Ascending Stairs | Dynamic | 1,544 | 1,073 | 471 |
| `3` | `2` | `WALKING_DOWNSTAIRS` | Descending Stairs | Dynamic | 1,406 | 986 | 420 |
| `4` | `3` | `SITTING` | Seated Posture | Static | 1,777 | 1,286 | 491 |
| `5` | `4` | `STANDING` | Upright Posture | Static | 1,906 | 1,374 | 532 |
| `6` | `5` | `LAYING` | Horizontal Supine/Prone | Static | 1,944 | 1,407 | 537 |

---

## 3. Participant Partitioning

- **Total Participants:** 30 volunteers (aged 19–48 years).
- **Training Group (21 Subjects, 70%):** `[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]`
- **Testing Group (9 Subjects, 30%):** `[2, 4, 9, 10, 12, 13, 18, 20, 24]`
- **Subject Leakage:** $0\%$ (Strict disjoint sets).

---

## 4. Processed Artifacts File Specifications

| Artifact Path | Shape | Data Type | Size on Disk | SHA-256 Checksum Prefix | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| `data/processed/X_train.npy` | `(7352, 128, 9)` | `float32` | 32.31 MB | `65be85ca36...` | Standardized triaxial sensor tensor for training |
| `data/processed/y_train.npy` | `(7352,)` | `int64` | 58.94 KB | `b9658087f2...` | 0-indexed training activity labels (0-5) |
| `data/processed/subject_train.npy` | `(7352,)` | `int64` | 58.94 KB | `66944a2403...` | Volunteer identifier per training window |
| `data/processed/X_test.npy` | `(2947, 128, 9)` | `float32` | 12.95 MB | `4b40755bc2...` | Standardized triaxial sensor tensor for testing |
| `data/processed/y_test.npy` | `(2947,)` | `int64` | 23.70 KB | `9d5716efc0...` | 0-indexed testing activity labels (0-5) |
| `data/processed/subject_test.npy` | `(2947,)` | `int64` | 23.70 KB | `533a077ec7...` | Volunteer identifier per testing window |
| `data/processed/label_mapping.json` | Key-Value Map | JSON | 464 Bytes | `d559bddbf5...` | Bidirectional mappings between raw IDs, target ints, and strings |
| `data/processed/preprocessing_metadata.json` | Object Schema | JSON | 2.15 KB | `f29ba189df...` | Complete pipeline parameters, scaler stats, and audit trails |
