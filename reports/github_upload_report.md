# GitHub Repository Upload & Verification Report

**Date & Time:** 2026-11-21  
**Project:** Human Activity Recognition from Wearable IMU Sensor Data using Time-Series Machine Learning  
**Local Repository Path:** `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml`  

---

## 1. GitHub Account & Authentication Verification

- **Authenticated GitHub Username:** `harishnarayanaan-prog`
- **Authenticated Email:** `harishnarayanaan@gmail.com`
- **Authentication Method:** GitHub CLI OAuth via browser device flow (`gh auth login --web`)
- **Keyring / Protocol:** HTTPS with system credential store
- **Token Scopes:** `gist`, `read:org`, `repo`

---

## 2. Remote Repository Details

- **Repository Name:** `imu-human-activity-recognition-ml`
- **GitHub Repository URL:** [https://github.com/harishnarayanaan-prog/imu-human-activity-recognition-ml](https://github.com/harishnarayanaan-prog/imu-human-activity-recognition-ml)
- **Clone URL (HTTPS):** `https://github.com/harishnarayanaan-prog/imu-human-activity-recognition-ml.git`
- **Visibility:** **PUBLIC** (Configured as an open internship portfolio repository)
- **Default Branch:** `main`
- **Initial Commit Hash:** `be6c33f8c6366a0bf19cfaf19a45c71b4d08fc3a`
- **Git Author & Committer Identity:**
  - Name: `Harish Narayanaan`
  - Email: `harishnarayanaan@gmail.com`

---

## 3. Uploaded Directory Structure & File Manifest

The GitHub repository contains all source code, test suites, documentation, figures, and pipeline metadata:

```
├── .gitignore
├── README.md
├── data/
│   ├── README.md
│   └── processed/
│       ├── label_mapping.json
│       └── preprocessing_metadata.json
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── environment_report.md
│   └── preprocessing_methodology.md
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── pytest.ini
├── reports/
│   ├── dataset_exploration.md
│   ├── day1_completion_report.md
│   ├── github_upload_report.md
│   └── figures/
│       ├── accelerometer_representative_activities.png
│       ├── activity_distribution.png
│       ├── gyroscope_representative_activities.png
│       ├── participant_distribution.png
│       ├── sensor_feature_correlation.png
│       ├── signal_amplitude_distributions.png
│       └── stationary_vs_dynamic_comparison.png
├── requirements.txt
├── requirements-dev.txt
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_dataset.py
│   │   ├── explore_dataset.py
│   │   ├── generate_metadata.py
│   │   ├── load_dataset.py
│   │   ├── preprocess.py
│   │   └── validate_dataset.py
│   └── simulation/
│       ├── __init__.py
│       └── synthetic_imu_generator.py
└── tests/
    ├── __init__.py
    ├── test_download.py
    ├── test_end_to_end_pipeline.py
    ├── test_loaders.py
    ├── test_preprocessing.py
    ├── test_simulation.py
    └── test_validation.py
```

---

## 4. Exclusion & Security Verification

Strict exclusion verification was performed using `git status --ignored` and GitHub Tree API queries:

| Category | Exclusion Status | Verification Details |
|:---|:---:|:---|
| **Virtual Environments** | **Excluded** | `.venv/` excluded via `.gitignore` |
| **Downloaded Raw Archives** | **Excluded** | `data/raw/*.zip` (61 MB archive) excluded |
| **Extracted Raw Files** | **Excluded** | `data/extracted/` excluded |
| **Large Processed Binary Tensors** | **Excluded** | `data/processed/*.npy` (45 MB) excluded |
| **Python Caches & Artifacts** | **Excluded** | `__pycache__/`, `.pytest_cache/`, `.matplotlib_cache/` excluded |
| **Credentials & Secrets** | **Excluded** | Exactly 0 `.env`, tokens, passwords, or personal keys |

---

## 5. Automated Test Suite Results

Full automated testing was re-executed prior to upload:
- **Framework:** `pytest 9.2.0` on Python 3.14.3
- **Total Tests:** 29
- **Tests Passed:** 29 (100%)
- **Tests Failed:** 0
- **Execution Time:** 10.17 seconds

---

## 6. Remote Repository Verification Summary

1. **Remote Origin:** Verified pointing to `https://github.com/harishnarayanaan-prog/imu-human-activity-recognition-ml.git`.
2. **Branch Tracking:** `main` set up to track `origin/main`.
3. **README Rendering:** Verified formatted markdown with badges, architecture flowcharts, empirical tables, and dataset citations.
4. **Remaining Issues:** **None.** The repository is cleanly published, fully reproducible, and ready for Day 2 development.
