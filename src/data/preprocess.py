"""Modular preprocessing pipeline for UCI HAR sensor tensors.

Applies label remapping (0-indexed standard), subject-wise splitting validation,
per-channel standardization fitted strictly on training data, shape and type assertions,
and serializes processed numpy arrays and metadata.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_dataset import (
    CHANNEL_NAMES,
    load_activity_labels,
    load_full_split,
    resolve_dataset_dir,
)
from src.data.validate_dataset import validate_splits

logger = logging.getLogger(__name__)


@dataclass
class ChannelScalerParams:
    """Stores per-channel normalization statistics computed on training set."""
    channel_name: str
    mean: float
    std: float


class ChannelWiseStandardScaler:
    """Scales 3D tensor (N, T, C) channels independently using training set statistics.

    Prevents data leakage by fitting exclusively on training partitions.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.means_: Optional[np.ndarray] = None  # Shape (C,)
        self.stds_: Optional[np.ndarray] = None   # Shape (C,)
        self.channel_names: List[str] = CHANNEL_NAMES

    def fit(self, X: np.ndarray) -> "ChannelWiseStandardScaler":
        """Compute mean and std per channel across all windows and timesteps.

        Args:
            X: Tensor of shape (samples, timesteps, channels).
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got shape {X.shape}")
        # Flatten samples and timesteps: (N * T, C)
        n_samples, n_steps, n_channels = X.shape
        flat = X.reshape(-1, n_channels)
        self.means_ = np.mean(flat, axis=0, dtype=np.float64)
        self.stds_ = np.std(flat, axis=0, dtype=np.float64)
        # Avoid division by zero
        self.stds_[self.stds_ < self.eps] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization: (X - mean) / std.

        Args:
            X: Tensor of shape (samples, timesteps, channels).
        """
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        # Broadcast across timesteps and samples
        return (X - self.means_) / self.stds_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Revert standardization: X * std + mean."""
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (X * self.stds_) + self.means_

    def get_params_dict(self) -> Dict[str, Dict[str, float]]:
        if self.means_ is None or self.stds_ is None:
            return {}
        return {
            ch: {"mean": float(self.means_[i]), "std": float(self.stds_[i])}
            for i, ch in enumerate(self.channel_names)
        }


def remap_labels_to_zero_indexed(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """Convert 1-indexed raw labels (1..6) to standard machine-learning 0-indexed labels (0..5)."""
    raw_unique = sorted(list(set(y.tolist())))
    mapping = {raw_lbl: idx for idx, raw_lbl in enumerate(raw_unique)}
    y_mapped = np.array([mapping[val] for val in y], dtype=np.int64)
    return y_mapped, mapping


def split_train_val_subject_wise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    val_subjects_count: int = 4,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform a strictly subject-wise train/validation split within the training partition.

    Guarantees no window from the same participant appears in both train and validation splits.
    """
    rng = np.random.RandomState(random_seed)
    unique_subjects = np.unique(subj_train)
    val_subjs = set(rng.choice(unique_subjects, size=val_subjects_count, replace=False))
    train_subjs = set(unique_subjects) - val_subjs

    val_mask = np.isin(subj_train, list(val_subjs))
    train_mask = ~val_mask

    return (
        X_train[train_mask], y_train[train_mask], subj_train[train_mask],
        X_train[val_mask], y_train[val_mask], subj_train[val_mask],
    )


def run_preprocessing_pipeline(
    dataset_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    apply_scaling: bool = True,
    dtype: np.dtype = np.float32,
    random_seed: int = 42,
) -> Dict:
    """Execute complete preprocessing pipeline and serialize artifacts."""
    base_dir = resolve_dataset_dir(dataset_dir)
    out_dir = Path(output_dir) if output_dir else ROOT_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting preprocessing pipeline from source: %s", base_dir)
    logger.info("Destination directory: %s", out_dir)

    # 1. Load raw data
    raw_activity_map = load_activity_labels(base_dir)
    X_train_raw, y_train_raw, subj_train, _ = load_full_split(base_dir, split="train", load_features=False)
    X_test_raw, y_test_raw, subj_test, _ = load_full_split(base_dir, split="test", load_features=False)

    # 2. Validate raw data
    val_report = validate_splits(X_train_raw, y_train_raw, subj_train, X_test_raw, y_test_raw, subj_test)
    if not val_report.is_valid:
        raise ValueError("Raw data validation failed: " + "; ".join(val_report.errors))

    # 3. Label Mapping (1..6 -> 0..5)
    y_train_mapped, raw_to_zero = remap_labels_to_zero_indexed(y_train_raw)
    y_test_mapped = np.array([raw_to_zero[val] for val in y_test_raw], dtype=np.int64)

    # Label dictionary for inference and evaluation
    # Maps 0-indexed int -> activity name
    zero_indexed_label_map = {
        int(raw_to_zero[raw_id]): name
        for raw_id, name in raw_activity_map.items()
    }

    # 4. Feature Scaling (Fit STRICTLY on X_train)
    scaler = ChannelWiseStandardScaler()
    if apply_scaling:
        logger.info("Fitting ChannelWiseStandardScaler on training partition...")
        X_train_processed = scaler.fit_transform(X_train_raw).astype(dtype)
        logger.info("Transforming test partition using training scaler statistics (no leakage)...")
        X_test_processed = scaler.transform(X_test_raw).astype(dtype)
        scaler_stats = scaler.get_params_dict()
    else:
        logger.info("Skipping scaling (exporting raw values)...")
        X_train_processed = X_train_raw.astype(dtype)
        X_test_processed = X_test_raw.astype(dtype)
        scaler_stats = {}

    # 5. Save numpy binary outputs
    logger.info("Saving processed numpy tensors...")
    np.save(out_dir / "X_train.npy", X_train_processed)
    np.save(out_dir / "y_train.npy", y_train_mapped)
    np.save(out_dir / "subject_train.npy", subj_train)

    np.save(out_dir / "X_test.npy", X_test_processed)
    np.save(out_dir / "y_test.npy", y_test_mapped)
    np.save(out_dir / "subject_test.npy", subj_test)

    # 6. Save label mapping
    label_map_file = out_dir / "label_mapping.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump({
            "raw_to_zero_index": {str(k): v for k, v in raw_to_zero.items()},
            "zero_index_to_activity": {str(k): v for k, v in zero_indexed_label_map.items()},
            "activity_to_zero_index": {v: k for k, v in zero_indexed_label_map.items()},
        }, f, indent=2)

    # 7. Metadata generation
    metadata = {
        "pipeline_version": "1.0.0",
        "random_seed": random_seed,
        "sampling_rate_hz": 50.0,
        "timesteps_per_window": 128,
        "window_duration_seconds": 2.56,
        "window_overlap_percent": 50.0,
        "num_channels": 9,
        "channel_names": CHANNEL_NAMES,
        "scaling_applied": apply_scaling,
        "scaling_method": "ChannelWiseStandardScaler (per-channel Z-score fitted on train)",
        "channel_scaler_parameters": scaler_stats,
        "data_type": str(dtype),
        "train_samples": len(X_train_processed),
        "test_samples": len(X_test_processed),
        "train_subjects": sorted(list(set(subj_train.tolist()))),
        "test_subjects": sorted(list(set(subj_test.tolist()))),
        "subject_leakage_verified": True,
        "artifacts_generated": [
            "X_train.npy",
            "y_train.npy",
            "subject_train.npy",
            "X_test.npy",
            "y_test.npy",
            "subject_test.npy",
            "label_mapping.json",
            "preprocessing_metadata.json",
        ],
    }

    meta_file = out_dir / "preprocessing_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Preprocessing complete. Metadata saved to: %s", meta_file)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    run_preprocessing_pipeline()
