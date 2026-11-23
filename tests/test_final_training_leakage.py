"""Regression tests verifying zero test-set leakage in final CNN training.

These tests prove:
1. CNN1DHAR.fit_fixed_epochs rejects validation/test arguments in its signature.
2. Final training does not calculate validation or test loss.
3. Final training does not use validation/test metrics for checkpoint or epoch selection.
4. The final model is reinitialized from scratch.
5. In run_corrected_final_training, CNN1DHAR.fit_fixed_epochs is called with predetermined
   fixed epochs, receiving strictly the training partition, while the test partition is
   passed exclusively to predict().
"""

from __future__ import annotations

import os
import tempfile
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.cnn_1d import CNN1DHAR, Conv1DNet
from src.models.random_forest import RandomForestHAR
from src.models.svm import SVMHAR


def test_fit_fixed_epochs_signature_rejects_validation_data() -> None:
    """Verify fit_fixed_epochs signature does not accept validation/test arguments."""
    sig = inspect.signature(CNN1DHAR.fit_fixed_epochs)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "X_train" in params
    assert "y_train" in params
    assert "epochs" in params
    assert "X_val" not in params, "fit_fixed_epochs must not accept X_val"
    assert "y_val" not in params, "fit_fixed_epochs must not accept y_val"
    assert "X_test" not in params, "fit_fixed_epochs must not accept X_test"
    assert "y_test" not in params, "fit_fixed_epochs must not accept y_test"

    cnn = CNN1DHAR(in_channels=9, num_classes=6)
    dummy_x = np.zeros((10, 128, 9), dtype=np.float32)
    dummy_y = np.zeros(10, dtype=np.int64)
    with pytest.raises(TypeError):
        # Passing unexpected validation keyword must immediately fail
        cnn.fit_fixed_epochs(dummy_x, dummy_y, epochs=1, X_val=dummy_x)  # type: ignore


def test_final_training_does_not_calculate_val_or_test_loss() -> None:
    """Verify that final training history has empty val_loss and val_acc."""
    cnn = CNN1DHAR(in_channels=9, num_classes=6, batch_size=8, random_seed=42)
    rng = np.random.RandomState(42)
    X = rng.randn(16, 128, 9).astype(np.float32)
    y = rng.choice([0, 1, 2, 3, 4, 5], size=16)

    cnn.fit_fixed_epochs(X, y, epochs=3)

    assert len(cnn.history_["train_loss"]) == 3
    assert len(cnn.history_["val_loss"]) == 0, "val_loss must be empty in fixed-epoch training"
    assert len(cnn.history_["val_acc"]) == 0, "val_acc must be empty in fixed-epoch training"
    assert cnn.best_epoch_ == 3, "best_epoch_ must match predetermined fixed epoch count"
    assert cnn.is_fitted_ is True


def test_final_model_reinitialized_from_scratch() -> None:
    """Verify fit_fixed_epochs resets network weights and does not reuse prior weights."""
    cnn = CNN1DHAR(in_channels=9, num_classes=6, batch_size=8, random_seed=42)
    rng = np.random.RandomState(42)
    X = rng.randn(16, 128, 9).astype(np.float32)
    y = rng.choice([0, 1, 2, 3, 4, 5], size=16)

    cnn.fit_fixed_epochs(X, y, epochs=2)
    weights_after_fit1 = {k: v.clone() for k, v in cnn.model.state_dict().items()}

    # Change weights manually
    with torch.no_grad():
        for p in cnn.model.parameters():
            p.add_(10.0)

    # Calling fit_fixed_epochs again with same seed must reset to exact initial state then fit
    cnn.fit_fixed_epochs(X, y, epochs=2)
    weights_after_fit2 = cnn.model.state_dict()

    for k in weights_after_fit1:
        assert torch.allclose(weights_after_fit1[k], weights_after_fit2[k], atol=1e-6), (
            f"Weights for {k} did not match across reproducible seed re-initialization"
        )


def test_pipeline_isolates_test_data_and_enforces_fixed_epochs(tmp_path: Path) -> None:
    """Verify that run_corrected_final_training passes strictly training data to fit

    and only passes test data to predict.
    """
    from scripts.run_corrected_pipeline import run_corrected_final_training

    rng = np.random.RandomState(42)
    N_train, N_test = 40, 20
    X_train_raw = rng.randn(N_train, 128, 9).astype(np.float32)
    y_train = rng.choice([0, 1, 2, 3, 4, 5], size=N_train)
    subj_train = rng.choice(range(1, 22), size=N_train)

    X_test_raw = rng.randn(N_test, 128, 9).astype(np.float32)
    y_test = rng.choice([0, 1, 2, 3, 4, 5], size=N_test)
    subj_test = rng.choice(range(22, 31), size=N_test)

    best_val_epochs = 24

    with patch("numpy.save"), \
         patch.object(RandomForestHAR, "save", autospec=True), \
         patch.object(SVMHAR, "save", autospec=True), \
         patch.object(CNN1DHAR, "save", autospec=True), \
         patch.object(pd.DataFrame, "to_csv", autospec=True), \
         patch.object(CNN1DHAR, "fit_fixed_epochs", autospec=True) as mock_fit, \
         patch.object(CNN1DHAR, "predict", autospec=True, return_value=np.zeros(N_test, dtype=np.int64)) as mock_predict:

        # Configure mock_fit to set attributes expected by pipeline
        def mock_fit_impl(self, X, y, epochs):
            self.training_time_sec_ = 1.0
            self.best_epoch_ = epochs
            self.is_fitted_ = True
            return self

        mock_fit.side_effect = mock_fit_impl

        # Run pipeline
        run_corrected_final_training(
            X_train_raw=X_train_raw,
            y_train=y_train,
            subj_train=subj_train,
            X_test_raw=X_test_raw,
            y_test=y_test,
            subj_test=subj_test,
            best_cnn_epochs=best_val_epochs,
            fig_dir=tmp_path,
        )

        # 1. Assert fit_fixed_epochs was called exactly once
        assert mock_fit.call_count == 1
        call_args = mock_fit.call_args

        # 2. Check positional arguments passed to fit_fixed_epochs: (self, X_train, y_train, epochs=...)
        passed_X_train = call_args[0][1]
        passed_y_train = call_args[0][2]
        passed_epochs = call_args[1].get("epochs", call_args[0][3] if len(call_args[0]) > 3 else None)

        assert len(passed_X_train) == N_train, "fit_fixed_epochs must receive training windows"
        assert len(passed_y_train) == N_train, "fit_fixed_epochs must receive training labels"
        assert passed_epochs == best_val_epochs, "fit_fixed_epochs must receive fixed epochs from validation"

        # 3. Assert test windows were NEVER passed to fit_fixed_epochs
        assert not np.array_equal(passed_X_train, X_test_raw)
        assert len(passed_X_train) != N_test

        # 4. Assert predict was called on test windows
        assert mock_predict.call_count == 1
        predict_X = mock_predict.call_args[0][1]
        assert len(predict_X) == N_test, "predict must receive test windows"


def test_validation_and_training_subjects_are_strictly_disjoint() -> None:
    """Verify that internal subject-wise split produces strictly disjoint participant sets."""
    from src.data.splitting import subject_wise_train_val_split

    rng = np.random.RandomState(42)
    subjects = np.array([1, 1, 3, 3, 5, 5, 11, 11, 17, 17, 27, 27, 28, 28])
    X = rng.randn(len(subjects), 128, 9).astype(np.float32)
    y = np.zeros(len(subjects), dtype=int)

    split = subject_wise_train_val_split(X, y, subjects, val_subjects=[3, 11, 17, 27])
    assert split.verify_disjointness() is True
    assert set(split.train_subjects).isdisjoint(set(split.val_subjects))
    assert split.summary()["is_strictly_disjoint"] is True


def test_scaler_isolated_to_internal_training_partition() -> None:
    """Verify that ChannelWiseStandardScaler fitted on internal train ignores validation samples."""
    from src.data.preprocess import ChannelWiseStandardScaler
    from src.data.splitting import prepare_leak_free_train_val_tensors

    X_raw = np.ones((10, 128, 9), dtype=np.float32)
    # Give validation subjects distinct values
    X_raw[6:] = 100.0
    y = np.zeros(10, dtype=int)
    subjects = np.array([1, 1, 5, 5, 6, 6, 3, 3, 11, 11])

    X_train, y_train, X_val, y_val, scaler, split = prepare_leak_free_train_val_tensors(
        X_raw, y, subjects, val_subjects=[3, 11]
    )

    # Scaler mean must equal 1.0 (internal training subjects [1, 5, 6]), NOT 100.0 or intermediate
    assert np.allclose(scaler.means_, 1.0)
    assert np.allclose(scaler.stds_, 1.0)  # std of constant 1.0 is 0.0, clipped to 1.0


def test_final_training_rejects_invalid_epoch_counts(tmp_path: Path) -> None:
    """Verify that run_corrected_final_training raises ValueError when given invalid epoch counts.

    Prohibits silent fallbacks to default epochs when validation fails.
    """
    from scripts.run_corrected_pipeline import run_corrected_final_training

    rng = np.random.RandomState(42)
    X = rng.randn(10, 128, 9).astype(np.float32)
    y = np.zeros(10, dtype=int)
    subjs = np.ones(10, dtype=int)

    invalid_epoch_values = [None, 0, -1, -32, "32", 0.5]

    for val in invalid_epoch_values:
        with pytest.raises(ValueError, match="Invalid validation-derived epoch count"):
            run_corrected_final_training(
                X_train_raw=X,
                y_train=y,
                subj_train=subjs,
                X_test_raw=X,
                y_test=y,
                subj_test=subjs,
                best_cnn_epochs=val,  # type: ignore
                fig_dir=tmp_path,
            )
