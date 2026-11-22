"""Unit tests for Random Forest, SVM, and 1D CNN models."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.random_forest import RandomForestHAR
from src.models.svm import SVMHAR
from src.models.cnn_1d import CNN1DHAR, Conv1DNet


def test_random_forest_fit_predict(tmp_path: Path) -> None:
    """Verify RandomForestHAR fits, predicts, computes importances, and saves/loads."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 204)
    y = rng.choice([0, 1, 2, 3, 4, 5], size=40)

    rf = RandomForestHAR(n_estimators=10, max_depth=5, random_state=42, n_jobs=1)
    rf.fit(X, y)

    preds = rf.predict(X[:5])
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1, 2, 3, 4, 5})

    probs = rf.predict_proba(X[:5])
    assert probs.shape == (5, 6)
    assert np.allclose(probs.sum(axis=1), 1.0)

    importances = rf.get_feature_importances()
    assert len(importances) == 204
    assert importances[0][1] >= importances[-1][1]

    save_file = tmp_path / "test_rf.joblib"
    rf.save(save_file)
    assert save_file.exists()

    loaded_rf = RandomForestHAR.load(save_file)
    loaded_preds = loaded_rf.predict(X[:5])
    assert np.array_equal(preds, loaded_preds)


def test_svm_fit_predict(tmp_path: Path) -> None:
    """Verify SVMHAR fits, predicts, and saves/loads."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 204)
    y = rng.choice([0, 1, 2, 3, 4, 5], size=30)

    svm = SVMHAR(C=1.0, kernel="rbf", probability=False, random_state=42)
    svm.fit(X, y)

    preds = svm.predict(X[:5])
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1, 2, 3, 4, 5})

    save_file = tmp_path / "test_svm.joblib"
    svm.save(save_file)
    assert save_file.exists()

    loaded_svm = SVMHAR.load(save_file)
    loaded_preds = loaded_svm.predict(X[:5])
    assert np.array_equal(preds, loaded_preds)


def test_cnn_1d_architecture() -> None:
    """Verify Conv1DNet forward pass tensor shapes."""
    net = Conv1DNet(in_channels=9, num_classes=6)
    dummy_input = torch.randn(4, 9, 128)
    output = net(dummy_input)
    assert output.shape == (4, 6)


def test_cnn_1d_fit_predict_save(tmp_path: Path) -> None:
    """Verify CNN1DHAR fit, prediction, parameter count, and serialization."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(20, 128, 9)
    y_train = rng.choice([0, 1, 2, 3, 4, 5], size=20)
    X_val = rng.randn(10, 128, 9)
    y_val = rng.choice([0, 1, 2, 3, 4, 5], size=10)

    cnn = CNN1DHAR(
        in_channels=9,
        num_classes=6,
        batch_size=10,
        max_epochs=2,
        patience=2,
        random_seed=42,
    )
    cnn.fit(X_train, y_train, X_val, y_val)

    assert cnn.count_parameters() > 50000

    preds = cnn.predict(X_val[:4])
    assert preds.shape == (4,)

    probs = cnn.predict_proba(X_val[:4])
    assert probs.shape == (4, 6)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    save_file = tmp_path / "test_cnn.pt"
    cnn.save(save_file)
    assert save_file.exists()

    loaded_cnn = CNN1DHAR.load(save_file, device="cpu")
    loaded_preds = loaded_cnn.predict(X_val[:4])
    assert np.array_equal(preds, loaded_preds)


def test_cnn_invalid_input_channels_raises_error() -> None:
    """Verify Conv1DNet raises RuntimeError when channel dimension does not match."""
    net = Conv1DNet(in_channels=9, num_classes=6)
    invalid_input = torch.randn(4, 3, 128)  # 3 channels instead of 9
    with pytest.raises(RuntimeError):
        net(invalid_input)


def test_models_unfitted_predict_raises_error() -> None:
    """Verify models raise descriptive exceptions when predicting before fit."""
    rf = RandomForestHAR(n_estimators=5)
    with pytest.raises(Exception):
        rf.predict(np.zeros((5, 204)))

    svm = SVMHAR()
    with pytest.raises(Exception):
        svm.predict(np.zeros((5, 204)))

    cnn = CNN1DHAR(in_channels=9, num_classes=6)
    with pytest.raises(Exception):
        cnn.predict(np.zeros((5, 128, 9)))
