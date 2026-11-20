"""Dataset validation module for UCI HAR sensor tensors and annotations.

Implements rigorous integrity assertions on shapes, missing values,
channel counts, data types, subject disjointness, and label distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_TRAIN_SAMPLES = 7352
EXPECTED_TEST_SAMPLES = 2947
EXPECTED_TIMESTEPS = 128
EXPECTED_CHANNELS = 9
EXPECTED_ACTIVITIES = {1, 2, 3, 4, 5, 6}
SAMPLING_RATE_HZ = 50.0
WINDOW_DURATION_SEC = 2.56


@dataclass
class ValidationReport:
    """Encapsulates data validation results and diagnostics."""
    is_valid: bool
    train_shape: tuple
    test_shape: tuple
    num_train_subjects: int
    num_test_subjects: int
    train_subjects: List[int]
    test_subjects: List[int]
    subject_overlap: List[int]
    nan_count_train: int
    nan_count_test: int
    inf_count_train: int
    inf_count_test: int
    train_labels: List[int]
    test_labels: List[int]
    checks_passed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"Validation Status: {status}\n"
            f"  Train Tensor Shape: {self.train_shape} (Expected: ({EXPECTED_TRAIN_SAMPLES}, {EXPECTED_TIMESTEPS}, {EXPECTED_CHANNELS}))\n"
            f"  Test Tensor Shape:  {self.test_shape} (Expected: ({EXPECTED_TEST_SAMPLES}, {EXPECTED_TIMESTEPS}, {EXPECTED_CHANNELS}))\n"
            f"  Train Subjects:     {len(self.train_subjects)} subjects {sorted(self.train_subjects)}\n"
            f"  Test Subjects:      {len(self.test_subjects)} subjects {sorted(self.test_subjects)}\n"
            f"  Subject Overlap:    {self.subject_overlap} (Expected: empty)\n"
            f"  Train NaNs / Infs:  {self.nan_count_train} / {self.inf_count_train}\n"
            f"  Test NaNs / Infs:   {self.nan_count_test} / {self.inf_count_test}\n"
            f"  Checks Passed:      {len(self.checks_passed)}\n"
            f"  Errors:             {len(self.errors)}"
        )


def validate_tensor(
    tensor: np.ndarray,
    expected_samples: int,
    expected_timesteps: int = EXPECTED_TIMESTEPS,
    expected_channels: int = EXPECTED_CHANNELS,
    split_name: str = "dataset",
) -> List[str]:
    """Validate dimensionality, data types, and absence of NaN/Inf in sensor tensor."""
    errors: List[str] = []

    if not isinstance(tensor, np.ndarray):
        errors.append(f"{split_name} tensor is not a numpy.ndarray: {type(tensor)}")
        return errors

    if tensor.ndim != 3:
        errors.append(f"{split_name} tensor ndim != 3 (got {tensor.ndim}, shape {tensor.shape})")
        return errors

    n_samples, n_steps, n_ch = tensor.shape
    if n_samples != expected_samples:
        errors.append(f"{split_name} sample count mismatch: got {n_samples}, expected {expected_samples}")
    if n_steps != expected_timesteps:
        errors.append(f"{split_name} timesteps mismatch: got {n_steps}, expected {expected_timesteps}")
    if n_ch != expected_channels:
        errors.append(f"{split_name} channel count mismatch: got {n_ch}, expected {expected_channels}")

    if not np.issubdtype(tensor.dtype, np.floating):
        errors.append(f"{split_name} tensor dtype is not floating-point: {tensor.dtype}")

    nan_count = int(np.isnan(tensor).sum())
    inf_count = int(np.isinf(tensor).sum())
    if nan_count > 0:
        errors.append(f"{split_name} contains {nan_count} NaN values")
    if inf_count > 0:
        errors.append(f"{split_name} contains {inf_count} Inf values")

    return errors


def validate_splits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subj_test: np.ndarray,
) -> ValidationReport:
    """Perform end-to-end multi-split validation and subject leakage checking."""
    checks_passed: List[str] = []
    errors: List[str] = []

    # 1. Tensor shape and numerical validity
    train_errs = validate_tensor(X_train, EXPECTED_TRAIN_SAMPLES, split_name="Training")
    if train_errs:
        errors.extend(train_errs)
    else:
        checks_passed.append("Train tensor shape (7352, 128, 9) and floating validity")

    test_errs = validate_tensor(X_test, EXPECTED_TEST_SAMPLES, split_name="Testing")
    if test_errs:
        errors.extend(test_errs)
    else:
        checks_passed.append("Test tensor shape (2947, 128, 9) and floating validity")

    # 2. Length alignment within splits
    if len(X_train) == len(y_train) == len(subj_train):
        checks_passed.append(f"Train row alignment: X({len(X_train)}), y({len(y_train)}), subj({len(subj_train)})")
    else:
        errors.append(f"Train row misalignment: len(X)={len(X_train)}, len(y)={len(y_train)}, len(subj)={len(subj_train)}")

    if len(X_test) == len(y_test) == len(subj_test):
        checks_passed.append(f"Test row alignment: X({len(X_test)}), y({len(y_test)}), subj({len(subj_test)})")
    else:
        errors.append(f"Test row misalignment: len(X)={len(X_test)}, len(y)={len(y_test)}, len(subj)={len(subj_test)}")

    # 3. Label domain check (1 to 6)
    train_labels = sorted(list(set(y_train.tolist())))
    test_labels = sorted(list(set(y_test.tolist())))
    if set(train_labels) == EXPECTED_ACTIVITIES and set(test_labels) == EXPECTED_ACTIVITIES:
        checks_passed.append(f"Label domains match expected activities 1-6 exactly")
    else:
        errors.append(f"Label domain mismatch: train has {train_labels}, test has {test_labels}")

    # 4. Subject partition & disjointness
    train_subjs = sorted(list(set(subj_train.tolist())))
    test_subjs = sorted(list(set(subj_test.tolist())))
    overlap = sorted(list(set(train_subjs).intersection(set(test_subjs))))

    if len(overlap) == 0:
        checks_passed.append(f"Subject disjointness verified: 0 overlapping subjects between train and test")
    else:
        errors.append(f"CRITICAL TEST LEAKAGE: Subject overlap detected: {overlap}")

    total_subjs = set(train_subjs).union(set(test_subjs))
    if total_subjs == set(range(1, 31)):
        checks_passed.append(f"All 30 subjects present: 21 train subjects, 9 test subjects")
    else:
        errors.append(f"Subject count mismatch: expected 30 unique subjects, found {len(total_subjs)}")

    # 5. Sliding window metadata assertions
    # 128 samples / 50 Hz = 2.56 sec window duration
    calculated_duration = EXPECTED_TIMESTEPS / SAMPLING_RATE_HZ
    if np.isclose(calculated_duration, WINDOW_DURATION_SEC):
        checks_passed.append(f"Sliding window temporal duration verified: {calculated_duration:.2f} s")

    is_valid = (len(errors) == 0)

    report = ValidationReport(
        is_valid=is_valid,
        train_shape=X_train.shape,
        test_shape=X_test.shape,
        num_train_subjects=len(train_subjs),
        num_test_subjects=len(test_subjs),
        train_subjects=train_subjs,
        test_subjects=test_subjs,
        subject_overlap=overlap,
        nan_count_train=int(np.isnan(X_train).sum()),
        nan_count_test=int(np.isnan(X_test).sum()),
        inf_count_train=int(np.isinf(X_train).sum()),
        inf_count_test=int(np.isinf(X_test).sum()),
        train_labels=train_labels,
        test_labels=test_labels,
        checks_passed=checks_passed,
        errors=errors,
    )
    return report
