"""Unit and integration tests for subject-wise dataset splitting."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.splitting import subject_wise_train_val_split, SubjectWiseKFold


def test_subject_wise_train_val_split_disjoint() -> None:
    """Verify train and val sets have zero overlapping subjects."""
    rng = np.random.RandomState(42)
    n_samples = 200
    X = rng.randn(n_samples, 20)
    y = rng.randint(0, 6, size=n_samples)
    subjects = rng.choice([1, 3, 5, 6, 7, 8, 11, 14, 15, 17], size=n_samples)

    result = subject_wise_train_val_split(
        X, y, subjects, val_subjects=[3, 11, 17]
    )

    assert result.verify_disjointness()
    overlap = set(result.train_subjects).intersection(set(result.val_subjects))
    assert len(overlap) == 0

    assert len(result.X_train) + len(result.X_val) == n_samples
    assert len(result.y_train) + len(result.y_val) == n_samples


def test_subject_wise_train_val_split_ratio() -> None:
    """Verify automated ratio-based subject selection."""
    rng = np.random.RandomState(42)
    n_samples = 300
    X = rng.randn(n_samples, 10)
    y = rng.randint(0, 6, size=n_samples)
    subjects = rng.choice(list(range(1, 21)), size=n_samples)

    result = subject_wise_train_val_split(
        X, y, subjects, val_ratio=0.2, random_seed=42
    )

    assert result.verify_disjointness()
    assert len(result.val_subjects) == 4  # 20% of 20 unique subjects


def test_subject_wise_kfold() -> None:
    """Verify SubjectWiseKFold yields completely disjoint groups across all folds."""
    rng = np.random.RandomState(42)
    n_samples = 150
    X = rng.randn(n_samples, 5)
    y = rng.randint(0, 6, size=n_samples)
    subjects = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=n_samples)

    cv = SubjectWiseKFold(n_splits=5)
    folds = list(cv.split(X, y, subjects))

    assert len(folds) == 5
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_subjs = set(subjects[train_idx])
        val_subjs = set(subjects[val_idx])
        assert len(train_subjs.intersection(val_subjs)) == 0, f"Fold {fold_idx} has subject overlap!"
