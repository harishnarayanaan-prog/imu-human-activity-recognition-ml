"""Feature engineering and extraction for IMU sensor time-series data.

Extracts statistical and frequency-domain features from 9-channel raw IMU windows
(body accelerometer, body gyroscope, total accelerometer) and 3 derived magnitude
channels, producing 204 engineered features per 128-sample window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

BASE_CHANNEL_NAMES: List[str] = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

MAGNITUDE_CHANNEL_NAMES: List[str] = [
    "body_acc_mag",
    "body_gyro_mag",
    "total_acc_mag",
]

ALL_CHANNEL_NAMES: List[str] = BASE_CHANNEL_NAMES + MAGNITUDE_CHANNEL_NAMES

FEATURE_TYPES: List[str] = [
    "mean",
    "std",
    "var",
    "min",
    "max",
    "median",
    "range",
    "rms",
    "mav",
    "iqr",
    "energy",
    "skewness",
    "kurtosis",
    "dominant_freq",
    "spectral_energy",
    "spectral_entropy",
    "frequency_centroid",
]


def compute_derived_magnitudes(X: np.ndarray) -> np.ndarray:
    """Compute 3 magnitude channels from 9-channel IMU windows.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, 128, 9) or (128, 9).

    Returns
    -------
    np.ndarray
        Array with 12 channels: original 9 + 3 magnitudes (body_acc_mag,
        body_gyro_mag, total_acc_mag).
    """
    is_2d = X.ndim == 2
    if is_2d:
        X = np.expand_dims(X, axis=0)

    body_acc_mag = np.linalg.norm(X[:, :, 0:3], axis=-1, keepdims=True)
    body_gyro_mag = np.linalg.norm(X[:, :, 3:6], axis=-1, keepdims=True)
    total_acc_mag = np.linalg.norm(X[:, :, 6:9], axis=-1, keepdims=True)

    X_full = np.concatenate([X, body_acc_mag, body_gyro_mag, total_acc_mag], axis=-1)

    if is_2d:
        return X_full[0]
    return X_full


def get_feature_names() -> List[str]:
    """Return ordered list of 204 engineered feature names."""
    names: List[str] = []
    for ch in ALL_CHANNEL_NAMES:
        for feat in FEATURE_TYPES:
            names.append(f"{ch}_{feat}")
    return names


def extract_features(
    X: np.ndarray,
    sampling_rate_hz: float = 50.0,
) -> np.ndarray:
    """Extract 204 time-series features from raw IMU windows.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, 128, 9) or (128, 9).
    sampling_rate_hz : float, default=50.0
        Sampling frequency in Hertz.

    Returns
    -------
    np.ndarray
        Extracted feature matrix of shape (N, 204) or (204,).
    """
    is_2d = X.ndim == 2
    if is_2d:
        X = np.expand_dims(X, axis=0)

    N, n_samples, n_channels = X.shape
    if n_channels != 9:
        raise ValueError(f"Expected 9 channels, got {n_channels}")
    if n_samples < 2:
        raise ValueError(f"Expected at least 2 samples per window, got {n_samples}")

    # Compute derived magnitudes (N, 128, 12)
    X_full = compute_derived_magnitudes(X)
    n_total_channels = X_full.shape[-1]  # 12

    # Pre-allocate feature matrix: shape (N, 12, 17)
    feats = np.zeros((N, n_total_channels, len(FEATURE_TYPES)), dtype=np.float64)

    # Statistical features (axis 1: time)
    mean = np.mean(X_full, axis=1)  # (N, 12)
    var = np.var(X_full, axis=1, ddof=1)
    std = np.std(X_full, axis=1, ddof=1)
    min_val = np.min(X_full, axis=1)
    max_val = np.max(X_full, axis=1)
    median_val = np.median(X_full, axis=1)
    range_val = max_val - min_val
    rms_val = np.sqrt(np.mean(X_full**2, axis=1))
    mav_val = np.mean(np.abs(X_full), axis=1)

    q75 = np.percentile(X_full, 75, axis=1)
    q25 = np.percentile(X_full, 25, axis=1)
    iqr_val = q75 - q25
    energy_val = np.mean(X_full**2, axis=1)

    # Skewness & Kurtosis with zero-variance protection
    diff = X_full - mean[:, None, :]
    safe_std = np.where(std > 1e-8, std, 1.0)
    skew_val = np.where(std > 1e-8, np.mean(diff**3, axis=1) / (safe_std**3), 0.0)
    kurt_val = np.where(std > 1e-8, (np.mean(diff**4, axis=1) / (safe_std**4)) - 3.0, 0.0)

    # Frequency-domain features using FFT
    rfft_vals = np.fft.rfft(X_full, axis=1)  # (N, n_freqs, 12)
    psd = np.abs(rfft_vals) ** 2
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate_hz)  # (n_freqs,)

    # Dominant frequency
    dom_idx = np.argmax(psd, axis=1)  # (N, 12)
    dom_freq = freqs[dom_idx]

    # Spectral energy
    spec_energy = np.sum(psd, axis=1) / float(n_samples)

    # Spectral entropy
    psd_sum = np.sum(psd, axis=1, keepdims=True)  # (N, 1, 12)
    n_freqs = len(freqs)
    psd_norm = np.where(psd_sum > 1e-12, psd / (psd_sum + 1e-12), 1.0 / float(n_freqs))
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=1)

    # Frequency centroid
    spec_centroid = np.sum(freqs[None, :, None] * psd, axis=1) / (
        np.squeeze(psd_sum, axis=1) + 1e-12
    )

    # Pack into (N, 12, 17)
    feat_list = [
        mean,
        std,
        var,
        min_val,
        max_val,
        median_val,
        range_val,
        rms_val,
        mav_val,
        iqr_val,
        energy_val,
        skew_val,
        kurt_val,
        dom_freq,
        spec_energy,
        spec_entropy,
        spec_centroid,
    ]

    for f_idx, f_arr in enumerate(feat_list):
        feats[:, :, f_idx] = f_arr

    # Reshape to (N, 204) where channel order is primary
    feats_flat = feats.reshape(N, -1)

    # Ensure no NaN or Inf
    feats_flat = np.nan_to_num(feats_flat, nan=0.0, posinf=1e6, neginf=-1e6)

    if is_2d:
        return feats_flat[0]
    return feats_flat


class IMUFeatureExtractor:
    """Scikit-Learn compatible feature extractor for IMU window time-series."""

    def __init__(self, sampling_rate_hz: float = 50.0):
        self.sampling_rate_hz = sampling_rate_hz
        self.feature_names_ = get_feature_names()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IMUFeatureExtractor":
        """Fit extractor (stateless, validates input)."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract 204 features from input windows."""
        return extract_features(X, sampling_rate_hz=self.sampling_rate_hz)

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and extract features."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Return names of extracted features."""
        return list(self.feature_names_)


def main() -> None:
    """Extract features from processed binary data and save output matrices."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    repo_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = repo_root / "data" / "processed"

    train_path = processed_dir / "X_train.npy"
    test_path = processed_dir / "X_test.npy"

    if not train_path.exists() or not test_path.exists():
        logger.error("Processed sensor arrays not found in %s", processed_dir)
        return

    logger.info("Loading processed sensor windows...")
    X_train = np.load(train_path)
    X_test = np.load(test_path)
    logger.info("X_train shape: %s, X_test shape: %s", X_train.shape, X_test.shape)

    extractor = IMUFeatureExtractor(sampling_rate_hz=50.0)

    logger.info("Extracting features from X_train...")
    X_train_features = extractor.transform(X_train)
    logger.info("Extracted X_train_features shape: %s", X_train_features.shape)

    logger.info("Extracting features from X_test...")
    X_test_features = extractor.transform(X_test)
    logger.info("Extracted X_test_features shape: %s", X_test_features.shape)

    out_train_path = processed_dir / "X_train_features.npy"
    out_test_path = processed_dir / "X_test_features.npy"
    np.save(out_train_path, X_train_features)
    np.save(out_test_path, X_test_features)
    logger.info("Saved feature matrices to %s and %s", out_train_path.name, out_test_path.name)

    metadata = {
        "n_samples_train": int(X_train_features.shape[0]),
        "n_samples_test": int(X_test_features.shape[0]),
        "n_features": int(X_train_features.shape[1]),
        "sampling_rate_hz": 50.0,
        "base_channels": BASE_CHANNEL_NAMES,
        "magnitude_channels": MAGNITUDE_CHANNEL_NAMES,
        "all_channels": ALL_CHANNEL_NAMES,
        "feature_types": FEATURE_TYPES,
        "feature_names": extractor.get_feature_names_out(),
    }

    metadata_path = processed_dir / "feature_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved feature metadata to %s", metadata_path.name)


if __name__ == "__main__":
    main()
