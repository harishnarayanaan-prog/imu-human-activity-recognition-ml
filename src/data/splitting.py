"""Subject-wise data splitting utilities for wearable IMU time-series.

Guarantees strict participant isolation between training and validation folds.
Prevents identity and temporal data leakage caused by window overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Container for subject-wise partitioned arrays and audit metadata."""
    X_train: np.ndarray
    y_train: np.ndarray
    subj_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    subj_val: np.ndarray
    train_subjects: List[int]
    val_subjects: List[int]
    train_indices: np.ndarray
    val_indices: np.ndarray

    def verify_disjointness(self) -> bool:
        """Verify that training and validation subjects do not overlap."""
        overlap = set(self.train_subjects).intersection(set(self.val_subjects))
        return len(overlap) == 0

    def summary(self) -> Dict:
        return {
            "n_train_samples": len(self.X_train),
            "n_val_samples": len(self.X_val),
            "train_subjects": sorted(self.train_subjects),
            "val_subjects": sorted(self.val_subjects),
            "train_subject_count": len(self.train_subjects),
            "val_subject_count": len(self.val_subjects),
            "overlap_count": len(set(self.train_subjects).intersection(set(self.val_subjects))),
            "is_strictly_disjoint": self.verify_disjointness(),
        }


def subject_wise_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    val_subjects: Optional[Sequence[int]] = None,
    n_val_subjects: int = 4,
    val_ratio: Optional[float] = None,
    random_seed: int = 42,
) -> SplitResult:
    """Split dataset windows into train and validation sets along whole subject boundaries.
    
    Args:
        X: Tensor or feature matrix of shape (N, ...).
        y: 1D array of activity labels (N,).
        subjects: 1D array of subject IDs (N,).
        val_subjects: Explicit sequence of subject IDs for validation. If None, chosen deterministically.
        n_val_subjects: Number of validation subjects if val_subjects and val_ratio are None.
        val_ratio: Fractional proportion of subjects for validation (e.g. 0.2 for 20%).
        random_seed: Seed for reproducible subject sampling.
        
    Returns:
        SplitResult with disjoint train and val partitions.
    """
    if len(X) != len(y) or len(X) != len(subjects):
        raise ValueError(f"Array length mismatch: X={len(X)}, y={len(y)}, subjects={len(subjects)}")

    unique_subjs = np.unique(subjects)

    if val_subjects is not None:
        val_set: Set[int] = set(val_subjects)
        missing = val_set - set(unique_subjs)
        if missing:
            raise ValueError(f"Specified validation subjects {missing} not found in dataset subjects {unique_subjs}")
        train_set: Set[int] = set(unique_subjs) - val_set
    elif val_ratio is not None:
        calc_n_val = max(1, int(round(len(unique_subjs) * val_ratio)))
        rng = np.random.RandomState(random_seed)
        shuffled = unique_subjs.copy()
        rng.shuffle(shuffled)
        val_set = set(shuffled[:calc_n_val])
        train_set = set(shuffled[calc_n_val:])
    else:
        rng = np.random.RandomState(random_seed)
        shuffled = unique_subjs.copy()
        rng.shuffle(shuffled)
        val_set = set(shuffled[:n_val_subjects])
        train_set = set(shuffled[n_val_subjects:])

    train_mask = np.isin(subjects, list(train_set))
    val_mask = np.isin(subjects, list(val_set))

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    result = SplitResult(
        X_train=X[train_idx],
        y_train=y[train_idx],
        subj_train=subjects[train_idx],
        X_val=X[val_idx],
        y_val=y[val_idx],
        subj_val=subjects[val_idx],
        train_subjects=sorted(list(train_set)),
        val_subjects=sorted(list(val_set)),
        train_indices=train_idx,
        val_indices=val_idx,
    )

    if not result.verify_disjointness():
        raise RuntimeError("CRITICAL ERROR: Subject overlap detected in subject_wise_train_val_split!")

    logger.info(
        "Subject-wise split generated: %d train windows (%d subjects) | %d val windows (%d subjects)",
        len(result.X_train), len(result.train_subjects),
        len(result.X_val), len(result.val_subjects),
    )
    return result


class SubjectWiseKFold:
    """Cross-validator generator yielding subject-disjoint train/val folds."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_seed: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.gkf = GroupKFold(n_splits=n_splits)

    def split(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray):
        """Yield (train_indices, val_indices) tuple for each fold with zero subject overlap."""
        return self.gkf.split(X, y, groups=subjects)
