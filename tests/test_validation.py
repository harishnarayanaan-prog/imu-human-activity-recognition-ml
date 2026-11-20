"""Unit tests for dataset integrity validation logic."""

import numpy as np
import pytest

from src.data.validate_dataset import (
    validate_tensor,
    validate_splits,
    EXPECTED_TRAIN_SAMPLES,
    EXPECTED_TEST_SAMPLES,
    EXPECTED_TIMESTEPS,
    EXPECTED_CHANNELS,
)


def test_validate_tensor_valid():
    """Test validate_tensor passes on well-formed tensor."""
    valid = np.zeros((100, 128, 9), dtype=np.float64)
    errs = validate_tensor(valid, expected_samples=100)
    assert len(errs) == 0


def test_validate_tensor_invalid_shape():
    """Test validate_tensor flags mismatched shapes and dimensions."""
    # 2D instead of 3D
    errs = validate_tensor(np.zeros((100, 128)), expected_samples=100)
    assert any("ndim != 3" in e for e in errs)

    # Wrong sample count
    errs = validate_tensor(np.zeros((50, 128, 9)), expected_samples=100)
    assert any("sample count mismatch" in e for e in errs)

    # Wrong channel count
    errs = validate_tensor(np.zeros((100, 128, 6)), expected_samples=100)
    assert any("channel count mismatch" in e for e in errs)


def test_validate_tensor_nan_and_inf_detection():
    """Test validate_tensor detects injected NaNs and Infs."""
    corrupt = np.zeros((10, 128, 9), dtype=np.float64)
    corrupt[0, 5, 2] = np.nan
    corrupt[1, 10, 3] = np.inf

    errs = validate_tensor(corrupt, expected_samples=10)
    assert any("NaN" in e for e in errs)
    assert any("Inf" in e for e in errs)


def test_validate_tensor_invalid_dtype():
    """Test validate_tensor rejects integer types for sensor signals."""
    int_tensor = np.zeros((10, 128, 9), dtype=np.int32)
    errs = validate_tensor(int_tensor, expected_samples=10)
    assert any("not floating-point" in e for e in errs)


def test_validate_splits_detects_subject_leakage():
    """Test validate_splits flags subject overlap between train and test."""
    X_tr = np.zeros((EXPECTED_TRAIN_SAMPLES, EXPECTED_TIMESTEPS, EXPECTED_CHANNELS), dtype=np.float64)
    X_te = np.zeros((EXPECTED_TEST_SAMPLES, EXPECTED_TIMESTEPS, EXPECTED_CHANNELS), dtype=np.float64)
    y_tr = np.random.randint(1, 7, size=EXPECTED_TRAIN_SAMPLES)
    y_te = np.random.randint(1, 7, size=EXPECTED_TEST_SAMPLES)

    # Subject 5 accidentally in both train and test
    sub_tr = np.ones(EXPECTED_TRAIN_SAMPLES, dtype=np.int64) * 5
    sub_te = np.ones(EXPECTED_TEST_SAMPLES, dtype=np.int64) * 5

    report = validate_splits(X_tr, y_tr, sub_tr, X_te, y_te, sub_te)
    assert not report.is_valid
    assert any("Subject overlap detected" in e for e in report.errors)
