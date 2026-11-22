"""Unit and integration tests for IMU feature engineering."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.feature_engineering import (
    IMUFeatureExtractor,
    extract_features,
    get_feature_names,
    compute_derived_magnitudes,
    ALL_CHANNEL_NAMES,
    FEATURE_TYPES,
)


def test_derived_magnitudes_shape() -> None:
    """Verify that magnitude computation expands 9 channels to 12 channels."""
    X_single = np.ones((128, 9), dtype=np.float64)
    X_out_single = compute_derived_magnitudes(X_single)
    assert X_out_single.shape == (128, 12)

    X_batch = np.ones((10, 128, 9), dtype=np.float64)
    X_out_batch = compute_derived_magnitudes(X_batch)
    assert X_out_batch.shape == (10, 128, 12)


def test_feature_names_count() -> None:
    """Verify feature name list has exactly 204 unique features."""
    names = get_feature_names()
    assert len(names) == 204
    assert len(names) == len(set(names)), "Feature names must be unique"
    assert len(ALL_CHANNEL_NAMES) == 12
    assert len(FEATURE_TYPES) == 17


def test_extract_features_single_window() -> None:
    """Verify feature extraction on a single (128, 9) window."""
    rng = np.random.RandomState(42)
    window = rng.randn(128, 9)
    feats = extract_features(window, sampling_rate_hz=50.0)

    assert feats.shape == (204,)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_extract_features_batch() -> None:
    """Verify feature extraction on a batch of windows (N, 128, 9)."""
    rng = np.random.RandomState(42)
    batch = rng.randn(5, 128, 9)
    feats = extract_features(batch, sampling_rate_hz=50.0)

    assert feats.shape == (5, 204)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_constant_signal_robustness() -> None:
    """Verify extractor does not produce NaNs or Infs on constant (zero-variance) signals."""
    constant_window = np.ones((128, 9), dtype=np.float64) * 3.5
    feats = extract_features(constant_window, sampling_rate_hz=50.0)

    assert feats.shape == (204,)
    assert not np.isnan(feats).any(), "Constant signal must not yield NaNs"
    assert not np.isinf(feats).any(), "Constant signal must not yield Infs"


def test_sklearn_transformer_interface() -> None:
    """Verify IMUFeatureExtractor works with scikit-learn transformer protocol."""
    rng = np.random.RandomState(42)
    X = rng.randn(4, 128, 9)

    extractor = IMUFeatureExtractor(sampling_rate_hz=50.0)
    feats = extractor.fit_transform(X)

    assert feats.shape == (4, 204)
    feature_names = extractor.get_feature_names_out()
    assert len(feature_names) == 204
