# UCI Human Activity Recognition Using Smartphones Dataset

## 1. Dataset Overview & Official Source

- **Repository:** UCI Machine Learning Repository
- **Dataset Name:** Human Activity Recognition Using Smartphones
- **Official URL:** [https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Direct Archive URL:** `https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip`
- **DOI:** [10.24432/C54S4K](https://doi.org/10.24432/C54S4K)
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

### Official Citation
```bibtex
@inproceedings{anguita2013public,
  title={A Public Domain Dataset for Human Activity Recognition Using Smartphones},
  author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge L},
  booktitle={21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013},
  year={2013}
}
```

---

## 2. Experimental Setup & Sensor Characteristics

- **Subjects:** 30 human participants (volunteers aged 19–48 years).
- **Device & Placement:** Samsung Galaxy S II smartphone mounted at waist level.
- **Embedded Sensors:** 3-axis accelerometer and 3-axis gyroscope.
- **Sampling Frequency:** 50 Hz (1 sample every 20 ms).
- **Window Segmentation:** Fixed-width sliding windows of 128 readings (2.56 seconds duration) with 50% overlap (64 samples step size).
- **Sensor Preprocessing Note:** **IMPORTANT:** The sensor recordings in this dataset are preprocessed, windowed inertial signals. They are **not** continuous, unsegmented raw sensor streams. The gravitational and body motion components were separated using a Butterworth low-pass filter (0.3 Hz corner frequency).

---

## 3. Inertial Signal Channels (9 Channels)

Each window consists of 128 time-series points across 9 distinct channels:
1. `body_acc_x`: Estimated triaxial body linear acceleration (X-axis, units in standard gravity $g$).
2. `body_acc_y`: Estimated triaxial body linear acceleration (Y-axis).
3. `body_acc_z`: Estimated triaxial body linear acceleration (Z-axis).
4. `body_gyro_x`: Triaxial angular velocity measured by gyroscope (X-axis, units in radians/second).
5. `body_gyro_y`: Triaxial angular velocity measured by gyroscope (Y-axis).
6. `body_gyro_z`: Triaxial angular velocity measured by gyroscope (Z-axis).
7. `total_acc_x`: Triaxial acceleration directly from the accelerometer device (X-axis, gravity + body).
8. `total_acc_y`: Triaxial acceleration directly from the accelerometer device (Y-axis).
9. `total_acc_z`: Triaxial acceleration directly from the accelerometer device (Z-axis).

---

## 4. Activity Classes

The dataset labels 6 primary human activities:

| ID | Activity Name | Activity Category | Description |
|:---|:---|:---|:---|
| 1 | `WALKING` | Dynamic | Level ground standard gait locomotion |
| 2 | `WALKING_UPSTAIRS` | Dynamic | Ascending stairs locomotion |
| 3 | `WALKING_DOWNSTAIRS`| Dynamic | Descending stairs locomotion |
| 4 | `SITTING` | Static | Upright stationary seated posture |
| 5 | `STANDING` | Static | Stationary upright standing posture |
| 6 | `LAYING` | Static | Horizontal resting posture (gravity predominantly on X-axis) |

---

## 5. Dataset Partitioning

The dataset was partitioned by subject to prevent subject-level data leakage:
- **Training Set (70% of subjects):** 21 subjects, 7,352 labeled windows.
- **Test Set (30% of subjects):** 9 subjects, 2,947 labeled windows.
- **Total Windows:** 10,299 windows.
- **Precomputed Feature Vectors:** 561 time and frequency domain features per window (stored in `X_train.txt` and `X_test.txt`).
- **Raw Window Tensors:** Shape `(7352, 128, 9)` for training and `(2947, 128, 9)` for testing.

---

## 6. Directory Layout & Integrity

```
data/
├── raw/
│   └── uci_har_dataset.zip        # 61,005,872 bytes archive from official UCI repository
├── extracted/
│   └── UCI HAR Dataset/
│       ├── activity_labels.txt    # 6 activity identifiers and textual names
│       ├── features.txt           # 561 feature column names
│       ├── features_info.txt      # Feature extraction methodology
│       ├── README.txt             # Original UCI publication README
│       ├── train/
│       │   ├── X_train.txt        # (7352, 561) engineered feature matrix
│       │   ├── y_train.txt        # (7352, 1) activity labels (1-6)
│       │   ├── subject_train.txt  # (7352, 1) subject IDs
│       │   └── Inertial Signals/  # 9 channel files: (7352, 128) each
│       └── test/
│           ├── X_test.txt         # (2947, 561) engineered feature matrix
│           ├── y_test.txt         # (2947, 1) activity labels (1-6)
│           ├── subject_test.txt   # (2947, 1) subject IDs
│           └── Inertial Signals/  # 9 channel files: (2947, 128) each
```

---

## 7. Dataset Limitations

1. **Windowed and Pre-filtered:** Data files are pre-segmented into 128-sample windows with fixed 50% overlap and pre-filtered using specific Butterworth filter cutoffs. They do not represent raw unsegmented continuous ADC readings.
2. **Single Sensor Location:** The smartphone was worn exclusively on the waist in a dedicated pouch. Motion characteristics differ significantly from modern smartwatch or wristband placements.
3. **Controlled Protocol:** Participants performed scripted sequences in a controlled laboratory environment rather than natural free-living daily behavior.
4. **Demographic Constraints:** Participants were between 19 and 48 years of age; gait patterns may differ from elderly or pediatric populations.
5. **Fixed Device Model:** Recorded on Samsung Galaxy S II smartphones; sensor bias, noise floor, and calibration specifics may vary on newer MEMS sensors.
