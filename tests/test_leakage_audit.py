"""Automated tests for data leakage prevention and methodology correctness."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.preprocess import ChannelWiseStandardScaler
from src.data.splitting import (
    subject_wise_train_val_split,
    prepare_leak_free_train_val_tensors,
    prepare_leak_free_train_val_features,
)
from src.models.random_forest import RandomForestHAR
from src.models.svm import SVMHAR
from src.models.cnn_1d import CNN1DHAR


def test_validation_participants_do_not_contribute_to_scaler() -> None:
    """Verify that validation participant data does NOT alter the scaler parameters."""
    rng = np.random.RandomState(42)
    n_train = 80
    n_val = 20
    n_total = n_train + n_val

    # Synthetic normal sensor signals
    X_clean = rng.randn(n_total, 128, 9)
    y = rng.randint(0, 6, size=n_total)
    subjects = np.array([1] * n_train + [99] * n_val)  # Subject 99 is validation

    # Normal split
    X_train_clean = X_clean[:n_train]
    expected_means = np.mean(X_train_clean.reshape(-1, 9), axis=0)
    expected_stds = np.std(X_train_clean.reshape(-1, 9), axis=0)

    # Contaminate validation subject with massive outlier values (+5000.0)
    X_contaminated = X_clean.copy()
    X_contaminated[n_train:] += 5000.0

    # Run leak-free pipeline
    (
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        scaler,
        split,
    ) = prepare_leak_free_train_val_tensors(
        X_contaminated, y, subjects, val_subjects=[99]
    )

    # Scaler must match clean internal training statistics, completely immune to validation outlier
    assert np.allclose(scaler.means_, expected_means, atol=1e-5), "Scaler mean was influenced by validation data!"
    assert np.allclose(scaler.stds_, expected_stds, atol=1e-5), "Scaler std was influenced by validation data!"


def test_training_and_validation_subjects_strictly_disjoint() -> None:
    """Verify train and val sets have zero overlapping subjects."""
    rng = np.random.RandomState(42)
    n_samples = 300
    X = rng.randn(n_samples, 128, 9)
    y = rng.randint(0, 6, size=n_samples)
    subjects = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=n_samples)

    (
        X_train_scaled,
        _,
        X_val_scaled,
        _,
        _,
        split,
    ) = prepare_leak_free_train_val_tensors(
        X, y, subjects, val_subjects=[3, 7, 10]
    )

    assert split.verify_disjointness()
    overlap = set(split.train_subjects).intersection(set(split.val_subjects))
    assert len(overlap) == 0, f"Detected subject overlap: {overlap}"


def test_validation_arrays_transformed_with_training_only_statistics() -> None:
    """Verify that validation arrays are scaled using strictly training means and stds."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 128, 9) * 2.5 + 1.5
    y = rng.randint(0, 6, size=50)
    subjects = np.array([1] * 40 + [2] * 10)

    (
        X_train_scaled,
        _,
        X_val_scaled,
        _,
        scaler,
        split,
    ) = prepare_leak_free_train_val_tensors(
        X, y, subjects, val_subjects=[2]
    )

    # Manual transformation of validation raw windows using training scaler
    expected_val_scaled = (split.X_val - scaler.means_) / scaler.stds_
    assert np.allclose(X_val_scaled, expected_val_scaled, atol=1e-6)


def test_official_test_transformed_with_training_only_statistics() -> None:
    """Verify official test partition uses official training statistics."""
    rng = np.random.RandomState(42)
    X_train_raw = rng.randn(60, 128, 9) * 3.0 + 0.5
    X_test_raw = rng.randn(30, 128, 9) * 1.5 - 2.0

    final_scaler = ChannelWiseStandardScaler()
    final_scaler.fit(X_train_raw)
    X_test_scaled = final_scaler.transform(X_test_raw)

    expected_test_scaled = (X_test_raw - final_scaler.means_) / final_scaler.stds_
    assert np.allclose(X_test_scaled, expected_test_scaled, atol=1e-6)

    # Ensure test data did not influence scaler
    test_flat = X_test_raw.reshape(-1, 9)
    test_means = np.mean(test_flat, axis=0)
    assert not np.allclose(final_scaler.means_, test_means)


def test_models_reproducibility_with_fixed_seed() -> None:
    """Verify that Random Forest and 1D CNN produce identical predictions with fixed random seeds."""
    rng = np.random.RandomState(42)
    X_feat = rng.randn(30, 204)
    y = rng.randint(0, 6, size=30)

    rf1 = RandomForestHAR(n_estimators=10, max_depth=5, random_state=42, n_jobs=1)
    rf1.fit(X_feat, y)
    p1 = rf1.predict(X_feat)

    rf2 = RandomForestHAR(n_estimators=10, max_depth=5, random_state=42, n_jobs=1)
    rf2.fit(X_feat, y)
    p2 = rf2.predict(X_feat)

    assert np.array_equal(p1, p2), "Random Forest must produce identical predictions with same seed!"

    # CNN reproducibility
    X_tensor = rng.randn(16, 128, 9)
    y_tensor = rng.randint(0, 6, size=16)

    torch.manual_seed(42)
    cnn1 = CNN1DHAR(in_channels=9, num_classes=6, batch_size=8, max_epochs=2, random_seed=42)
    cnn1.fit(X_tensor, y_tensor, X_tensor[:4], y_tensor[:4])
    cp1 = cnn1.predict(X_tensor[:4])

    torch.manual_seed(42)
    cnn2 = CNN1DHAR(in_channels=9, num_classes=6, batch_size=8, max_epochs=2, random_seed=42)
    cnn2.fit(X_tensor, y_tensor, X_tensor[:4], y_tensor[:4])
    cp2 = cnn2.predict(X_tensor[:4])

    assert np.array_equal(cp1, cp2), "1D CNN must produce identical predictions with same seed!"
