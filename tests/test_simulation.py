"""Unit tests for synthetic IMU generator."""

import numpy as np
import pytest

from src.simulation.synthetic_imu_generator import (
    SUPPORTED_ACTIVITIES,
    SyntheticIMUConfig,
    SyntheticIMUGenerator,
)


def test_generator_default_config():
    """Test generator initializes with valid default parameters."""
    config = SyntheticIMUConfig(sampling_rate_hz=50.0, duration_sec=2.56)
    assert config.timesteps == 128


def test_generate_stationary_window():
    """Test generating stationary postures produces correct shapes and synthetic tags."""
    gen = SyntheticIMUGenerator(SyntheticIMUConfig(random_seed=42))
    for act in ["STANDING", "SITTING", "LAYING"]:
        out = gen.generate_stationary_window(act)
        assert out["time"].shape == (128,)
        assert out["body_acc"].shape == (128, 3)
        assert out["total_acc"].shape == (128, 3)
        assert out["gyro"].shape == (128, 3)
        assert out["metadata"]["is_synthetic"] is True
        assert out["metadata"]["activity"] == act


def test_generate_dynamic_window():
    """Test generating gait locomotion produces valid waveforms and synthetic tags."""
    gen = SyntheticIMUGenerator(SyntheticIMUConfig(random_seed=42))
    for act in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
        out = gen.generate_dynamic_window(act)
        assert out["time"].shape == (128,)
        assert out["body_acc"].shape == (128, 3)
        assert out["total_acc"].shape == (128, 3)
        assert out["gyro"].shape == (128, 3)
        assert out["metadata"]["is_synthetic"] is True
        assert out["metadata"]["activity"] == act


def test_generate_unsupported_activity_raises():
    """Test that requesting an invalid activity name raises ValueError."""
    gen = SyntheticIMUGenerator()
    with pytest.raises(ValueError, match="Unsupported activity"):
        gen.generate_window("FLYING")


def test_generate_batch_dataset():
    """Test batch dataset generation format matches (N, 128, 9)."""
    gen = SyntheticIMUGenerator(SyntheticIMUConfig(random_seed=100))
    X, y, meta = gen.generate_batch_dataset(samples_per_activity=5)

    assert X.shape == (30, 128, 9)  # 6 activities * 5 = 30
    assert len(y) == 30
    assert meta["is_synthetic"] is True
    assert "WARNING" in meta
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
