# Windows Environment Inspection & Setup Report

**Date & Time:** 2026-11-21
**Project:** Human Activity Recognition from Wearable IMU Sensor Data using Time-Series Machine Learning
**Workspace:** `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml`

---

## 1. Operating System & Hardware Platform

| Attribute | Detected Value | Notes |
|:---|:---|:---|
| **Operating System** | Windows 11 Pro / Enterprise (NT 10.0.26200.0) | 64-bit architecture (x86_64) |
| **Shell** | PowerShell 5.1 / 7+ Core compatible | Native Windows execution environment |
| **Available Drives & Disk Space** | `C:\`: 231.57 GB Total, 16.73 GB Free<br>`D:\`: 781.50 GB Total, 466.83 GB Free | Workspace safely located on `D:\` with >460 GB free storage |

---

## 2. Developer Tooling & Compilers

| Tool | Status | Path / Version |
|:---|:---|:---|
| **Python** | Available | `Python 3.14.3` (`C:\Python314\python.exe`) |
| **Pip (System)** | Available | `pip 26.2.1` |
| **Git** | Available | `git version 2.56.1.windows.1` |
| **Git Configuration** | Configured | `user.name = harishnarayanaan-prog`<br>`user.email = harishnarayanaan@gmail.com`<br>`init.defaultbranch = master` |
| **VS Code** | Available | `C:\Users\LENOVO\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd` |
| **Jupyter** | Available | `C:\Users\LENOVO\AppData\Roaming\Python\Python314\Scripts\jupyter.exe` |
| **C/C++ Compiler** | Available | `gcc.exe` (`D:\mingw64\bin\gcc.exe`) |
| **Internet Connectivity** | Verified Active | Successfully verified live connection to `https://archive.ics.uci.edu` (HTTP 200) |

---

## 3. Global vs. Isolated Virtual Environment

Initial global environment inspection showed missing machine-learning libraries (`scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`). Per project rules, global packages were kept untouched and an isolated project-specific virtual environment was initialized:

- **Virtual Environment Path:** `D:\Projects\Wearable-Sensor-Platform\imu-human-activity-recognition-ml\.venv`
- **Python Binary:** `.venv\Scripts\python.exe`
- **Pip In Venv:** Upgraded to `26.3`

### Installed Dependency Versions (.venv)

| Package | Specified Minimum | Installed Version | Role |
|:---|:---|:---|:---|
| **numpy** | `>=2.0.0` | `2.5.5` | High-performance multidimensional array & tensor processing |
| **pandas** | `>=2.2.0` | `3.1.1` | Tabular metadata, subject & activity indexing |
| **scipy** | `>=1.15.0` | `1.18.1` | Signal filtering, spectral analysis, statistical routines |
| **scikit-learn** | `>=1.5.0` | `1.9.1` | Preprocessing scalers, baseline models, stratified metrics |
| **matplotlib** | `>=3.9.0` | `3.11.2` | Core static sensor signal visualization |
| **seaborn** | `>=0.13.0` | `0.13.2` | Statistical distributions & correlation heatmaps |
| **requests** | `>=2.31.0` | `2.34.2` | Robust dataset download streaming with checksums |
| **joblib** | `>=1.4.0` | `1.6.0` | Model and preprocessing artifact serialization |
| **pytest** | `>=8.0.0` | `9.2.0` | Automated testing framework |
| **jupyter** | `>=1.0.0` | `1.1.1` | Interactive exploratory analysis notebooks |

---

## 4. Verification Check

All libraries import cleanly and report valid versions in `.venv`. Dependency specifications were recorded in `requirements.txt` and `requirements-dev.txt`.
