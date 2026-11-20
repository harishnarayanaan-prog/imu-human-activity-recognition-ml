"""Modular dataset loader for UCI Human Activity Recognition (HAR) dataset.

Provides utilities to read tabular feature sets, subject IDs, activity labels,
and triaxial inertial sensor time-series channels (accelerometer and gyroscope)
into structured NumPy arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

CHANNEL_NAMES = [
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


def resolve_dataset_dir(dataset_dir: Optional[Path | str] = None) -> Path:
    """Resolve the path to the extracted UCI HAR Dataset directory."""
    if dataset_dir is not None:
        p = Path(dataset_dir)
        if p.name != "UCI HAR Dataset" and (p / "UCI HAR Dataset").is_dir():
            return p / "UCI HAR Dataset"
        return p

    # Standard workspace location
    root = Path(__file__).resolve().parents[2]
    cand = root / "data" / "extracted" / "UCI HAR Dataset"
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f"UCI HAR Dataset not found at standard path: {cand}")


def load_activity_labels(dataset_dir: Optional[Path | str] = None) -> Dict[int, str]:
    """Load activity identifier to string name mapping.

    Returns:
        Dict mapping integer ID (1-6) to uppercase activity name.
    """
    base = resolve_dataset_dir(dataset_dir)
    lbl_file = base / "activity_labels.txt"
    if not lbl_file.is_file():
        raise FileNotFoundError(f"Activity labels file not found: {lbl_file}")

    mapping: Dict[int, str] = {}
    with open(lbl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            mapping[int(parts[0])] = parts[1]
    return mapping


def load_feature_names(dataset_dir: Optional[Path | str] = None) -> List[str]:
    """Load the list of 561 precomputed feature names.

    Returns:
        List of 561 string feature names.
    """
    base = resolve_dataset_dir(dataset_dir)
    feat_file = base / "features.txt"
    if not feat_file.is_file():
        raise FileNotFoundError(f"Features file not found: {feat_file}")

    features: List[str] = []
    with open(feat_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            features.append(parts[1] if len(parts) > 1 else parts[0])
    return features


def load_labels(dataset_dir: Optional[Path | str] = None, split: str = "train") -> np.ndarray:
    """Load 1D array of activity labels for the given split.

    Args:
        dataset_dir: Path to dataset directory.
        split: 'train' or 'test'.

    Returns:
        1D int64 array of shape (N,) with label values in [1, 6].
    """
    base = resolve_dataset_dir(dataset_dir)
    y_file = base / split / f"y_{split}.txt"
    if not y_file.is_file():
        raise FileNotFoundError(f"Label file not found: {y_file}")

    y = np.loadtxt(y_file, dtype=np.int64)
    return y.reshape(-1)


def load_subjects(dataset_dir: Optional[Path | str] = None, split: str = "train") -> np.ndarray:
    """Load 1D array of subject (volunteer) IDs for the given split.

    Args:
        dataset_dir: Path to dataset directory.
        split: 'train' or 'test'.

    Returns:
        1D int64 array of shape (N,) with subject IDs in [1, 30].
    """
    base = resolve_dataset_dir(dataset_dir)
    subj_file = base / split / f"subject_{split}.txt"
    if not subj_file.is_file():
        raise FileNotFoundError(f"Subject file not found: {subj_file}")

    subj = np.loadtxt(subj_file, dtype=np.int64)
    return subj.reshape(-1)


def load_feature_matrix(dataset_dir: Optional[Path | str] = None, split: str = "train") -> np.ndarray:
    """Load the 561 precomputed tabular features matrix.

    Args:
        dataset_dir: Path to dataset directory.
        split: 'train' or 'test'.

    Returns:
        2D float64 array of shape (N, 561).
    """
    base = resolve_dataset_dir(dataset_dir)
    x_file = base / split / f"X_{split}.txt"
    if not x_file.is_file():
        raise FileNotFoundError(f"Feature matrix file not found: {x_file}")

    logger.info("Loading feature matrix %s...", x_file)
    X = np.loadtxt(x_file, dtype=np.float64)
    return X


def load_inertial_signals(
    dataset_dir: Optional[Path | str] = None,
    split: str = "train",
    channels: Optional[List[str]] = None,
) -> np.ndarray:
    """Load the 9 windowed inertial signal channels into a 3D tensor.

    Output tensor structure:
        (samples, timesteps, channels) = (N, 128, 9)

    Args:
        dataset_dir: Path to dataset directory.
        split: 'train' or 'test'.
        channels: Optional custom list of channels; defaults to CHANNEL_NAMES.

    Returns:
        3D float64 array of shape (N, 128, len(channels)).
    """
    base = resolve_dataset_dir(dataset_dir)
    target_channels = channels or CHANNEL_NAMES
    sig_dir = base / split / "Inertial Signals"
    if not sig_dir.is_dir():
        raise FileNotFoundError(f"Inertial Signals directory not found: {sig_dir}")

    channel_arrays: List[np.ndarray] = []
    for ch in target_channels:
        ch_file = sig_dir / f"{ch}_{split}.txt"
        if not ch_file.is_file():
            raise FileNotFoundError(f"Inertial channel file not found: {ch_file}")
        arr = np.loadtxt(ch_file, dtype=np.float64)  # Shape (N, 128)
        channel_arrays.append(arr)

    # Stack along third axis: (N, 128, num_channels)
    tensor = np.stack(channel_arrays, axis=-1)
    logger.info("Loaded inertial tensor for '%s': shape %s", split, tensor.shape)
    return tensor


def load_full_split(
    dataset_dir: Optional[Path | str] = None,
    split: str = "train",
    load_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load signals tensor, labels, subjects, and optionally engineered feature matrix.

    Returns:
        (X_signals, y, subjects, X_features)
    """
    signals = load_inertial_signals(dataset_dir, split=split)
    labels = load_labels(dataset_dir, split=split)
    subjects = load_subjects(dataset_dir, split=split)
    features = load_feature_matrix(dataset_dir, split=split) if load_features else None
    return signals, labels, subjects, features
