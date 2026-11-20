"""Unit tests for preprocessing, scaling, and subject-wise partitioning."""

import numpy as np
import pytest

from src.data.preprocess import (
    ChannelWiseStandardScaler,
    remap_labels_to_zero_indexed,
    split_train_val_subject_wise,
)


def test_channel_wise_standard_scaler_train_fit():
    """Test ChannelWiseStandardScaler normalizes channels to mean=0, std=1."""
    # Synthetic random tensor with known per-channel offsets
    rng = np.random.RandomState(42)
    X = rng.normal(loc=5.0, scale=2.0, size=(100, 128, 9))

    scaler = ChannelWiseStandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check flattened per-channel mean and std
    flat_scaled = X_scaled.reshape(-1, 9)
    means = np.mean(flat_scaled, axis=0)
    stds = np.std(flat_scaled, axis=0)

    np.testing.assert_allclose(means, 0.0, atol=1e-5)
    np.testing.assert_allclose(stds, 1.0, atol=1e-5)


def test_channel_wise_standard_scaler_no_test_leakage():
    """Verify test transformation uses train parameters without updating them."""
    X_train = np.ones((50, 128, 9)) * 10.0
    X_test = np.ones((20, 128, 9)) * 20.0

    scaler = ChannelWiseStandardScaler()
    scaler.fit(X_train)

    train_means = scaler.means_.copy()
    X_test_scaled = scaler.transform(X_test)

    # Ensure train scaler parameters remained unchanged
    np.testing.assert_array_equal(scaler.means_, train_means)
    # With std 1.0 (constant variance fallback) and mean 10.0: (20.0 - 10.0) / 1.0 = 10.0
    np.testing.assert_allclose(X_test_scaled, 10.0)


def test_channel_wise_standard_scaler_inverse_transform():
    """Test that inverse_transform accurately recovers original values."""
    rng = np.random.RandomState(123)
    X = rng.uniform(-5.0, 5.0, size=(30, 128, 9))

    scaler = ChannelWiseStandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)

    np.testing.assert_allclose(X_recovered, X, atol=1e-5)


def test_remap_labels_to_zero_indexed():
    """Test converting 1-indexed (1..6) to 0-indexed (0..5)."""
    y_raw = np.array([1, 2, 3, 4, 5, 6, 1, 6], dtype=np.int64)
    y_mapped, mapping = remap_labels_to_zero_indexed(y_raw)

    assert set(y_mapped.tolist()) == {0, 1, 2, 3, 4, 5}
    assert mapping[1] == 0
    assert mapping[6] == 5
    assert len(y_mapped) == len(y_raw)


def test_split_train_val_subject_wise():
    """Test subject-wise validation split creates strictly disjoint subject sets."""
    # 10 subjects: 1 to 10
    subjects = np.repeat(np.arange(1, 11), 20)  # 200 windows total
    X = np.zeros((200, 128, 9))
    y = np.ones(200, dtype=np.int64)

    X_tr, y_tr, sub_tr, X_val, y_val, sub_val = split_train_val_subject_wise(
        X, y, subjects, val_subjects_count=3, random_seed=42
    )

    tr_subjs = set(sub_tr.tolist())
    val_subjs = set(sub_val.tolist())

    assert len(val_subjs) == 3
    assert len(tr_subjs) == 7
    # Crucial assertion: zero subject overlap
    assert len(tr_subjs.intersection(val_subjs)) == 0
    assert len(X_tr) + len(X_val) == 200
