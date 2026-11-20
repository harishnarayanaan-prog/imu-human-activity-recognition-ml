"""Synthetic IMU sensor signal generator for pipeline testing and verification.

Generates physically plausible triaxial accelerometer and gyroscope time-series
for stationary postures (sitting, standing, laying) and dynamic locomotion
(walking, ascending stairs, descending stairs) with configurable sensor physics
(noise, bias, drift, cadence, and sampling frequency).

CRITICAL NOTE:
All generated outputs are explicitly tagged as `SYNTHETIC`.
Synthetic data is intended exclusively for software unit tests and pipeline validation;
it must never be mixed with empirical UCI HAR sensor data during model evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


@dataclass
class SyntheticIMUConfig:
    """Configuration parameters for synthetic IMU generation."""
    sampling_rate_hz: float = 50.0
    duration_sec: float = 2.56
    noise_std_acc: float = 0.02    # Gaussian noise std (units: g)
    noise_std_gyro: float = 0.05   # Gaussian noise std (units: rad/s)
    bias_acc: Tuple[float, float, float] = (0.01, -0.01, 0.02)
    bias_gyro: Tuple[float, float, float] = (0.005, -0.008, 0.003)
    drift_rate_acc: float = 0.0005 # Linear drift per second
    drift_rate_gyro: float = 0.001
    random_seed: Optional[int] = 42

    @property
    def timesteps(self) -> int:
        return int(round(self.sampling_rate_hz * self.duration_sec))


class SyntheticIMUGenerator:
    """Simulates triaxial accelerometer and gyroscope time series."""

    def __init__(self, config: Optional[SyntheticIMUConfig] = None):
        self.config = config or SyntheticIMUConfig()
        self.rng = np.random.RandomState(self.config.random_seed)

    def _generate_time_vector(self, duration_sec: Optional[float] = None) -> np.ndarray:
        dur = duration_sec or self.config.duration_sec
        n_points = int(round(self.config.sampling_rate_hz * dur))
        return np.linspace(0.0, dur, n_points, endpoint=False)

    def _add_sensor_imperfections(
        self,
        signals: np.ndarray,
        noise_std: float,
        bias: Tuple[float, float, float],
        drift_rate: float,
        t: np.ndarray,
    ) -> np.ndarray:
        """Inject additive zero-mean Gaussian noise, constant DC bias, and linear drift."""
        n_steps, _ = signals.shape
        noise = self.rng.normal(0.0, noise_std, size=signals.shape)
        bias_arr = np.array(bias, dtype=np.float64)
        drift = np.outer(t * drift_rate, np.ones(3))
        return signals + noise + bias_arr + drift

    def generate_stationary_window(
        self,
        activity: str = "STANDING",
        duration_sec: Optional[float] = None,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """Simulate stationary posture window.

        Gravity orientation varies by posture:
        - STANDING: vertical axis receives ~1.0 g gravity, slight sway.
        - SITTING: vertical axis receives ~0.85-0.95 g, forward tilt.
        - LAYING: vertical axis near 0.0 g, lateral/longitudinal axis receives ~1.0 g.
        """
        t = self._generate_time_vector(duration_sec)
        n = len(t)

        # Baseline body acceleration (low-frequency micro-tremor: 8-12 Hz physiological tremor)
        tremor_freq = 9.0
        tremor_amp = 0.005
        body_acc = np.zeros((n, 3))
        for c in range(3):
            body_acc[:, c] = tremor_amp * np.sin(2 * np.pi * tremor_freq * t + self.rng.uniform(0, np.pi))

        # Gravity orientation
        if activity == "STANDING":
            gravity = np.array([0.98, 0.05, 0.02])
        elif activity == "SITTING":
            gravity = np.array([0.88, 0.12, 0.04])
        elif activity == "LAYING":
            gravity = np.array([-0.15, 0.95, 0.20])
        else:
            gravity = np.array([0.95, 0.05, 0.05])

        total_acc_clean = body_acc + gravity
        total_acc = self._add_sensor_imperfections(
            total_acc_clean,
            self.config.noise_std_acc,
            self.config.bias_acc,
            self.config.drift_rate_acc,
            t,
        )

        # Gyroscope: very low angular velocity in stationary states
        gyro_clean = self.rng.normal(0.0, 0.005, size=(n, 3))
        gyro = self._add_sensor_imperfections(
            gyro_clean,
            self.config.noise_std_gyro,
            self.config.bias_gyro,
            self.config.drift_rate_gyro,
            t,
        )

        return {
            "time": t,
            "body_acc": body_acc,
            "total_acc": total_acc,
            "gyro": gyro,
            "metadata": {
                "is_synthetic": True,
                "data_source": "SYNTHETIC_IMU_GENERATOR",
                "activity": activity,
                "duration_sec": t[-1] + (t[1] - t[0]),
                "sampling_rate_hz": self.config.sampling_rate_hz,
            },
        }

    def generate_dynamic_window(
        self,
        activity: str = "WALKING",
        cadence_hz: float = 1.8,
        duration_sec: Optional[float] = None,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """Simulate dynamic gait locomotion window (walking, upstairs, downstairs)."""
        t = self._generate_time_vector(duration_sec)
        n = len(t)

        # Activity-dependent gait kinematics
        if activity == "WALKING":
            freq = cadence_hz
            vert_amp = 0.35
            ap_amp = 0.20
            lat_amp = 0.12
            gyro_amp = 1.2
            gravity = np.array([0.90, 0.08, 0.03])
        elif activity == "WALKING_UPSTAIRS":
            freq = cadence_hz * 0.9
            vert_amp = 0.50  # Stronger vertical push-off
            ap_amp = 0.28
            lat_amp = 0.15
            gyro_amp = 1.6
            gravity = np.array([0.85, 0.15, 0.05])
        elif activity == "WALKING_DOWNSTAIRS":
            freq = cadence_hz * 1.1
            vert_amp = 0.65  # Higher impact shock on step impact
            ap_amp = 0.30
            lat_amp = 0.18
            gyro_amp = 1.8
            gravity = np.array([0.88, 0.12, 0.04])
        else:
            freq = cadence_hz
            vert_amp, ap_amp, lat_amp, gyro_amp = 0.35, 0.20, 0.12, 1.2
            gravity = np.array([0.90, 0.08, 0.03])

        # Synthesize harmonic body acceleration
        phi = self.rng.uniform(0, 2 * np.pi)
        body_x = vert_amp * np.sin(2 * np.pi * freq * t + phi) + 0.15 * np.sin(4 * np.pi * freq * t + phi)
        body_y = ap_amp * np.cos(2 * np.pi * freq * t + phi)
        body_z = lat_amp * np.sin(2 * np.pi * (freq / 2.0) * t + phi)  # Lateral sway at half-stride rate
        body_acc = np.stack([body_x, body_y, body_z], axis=-1)

        total_acc_clean = body_acc + gravity
        total_acc = self._add_sensor_imperfections(
            total_acc_clean,
            self.config.noise_std_acc,
            self.config.bias_acc,
            self.config.drift_rate_acc,
            t,
        )

        # Gyroscope rotations: pelvic tilt, pitch, and yaw during gait
        gyro_x = gyro_amp * 0.4 * np.sin(2 * np.pi * freq * t + phi)
        gyro_y = gyro_amp * 0.8 * np.cos(2 * np.pi * freq * t + phi)
        gyro_z = gyro_amp * 0.5 * np.sin(2 * np.pi * (freq / 2.0) * t + phi)
        gyro_clean = np.stack([gyro_x, gyro_y, gyro_z], axis=-1)
        gyro = self._add_sensor_imperfections(
            gyro_clean,
            self.config.noise_std_gyro,
            self.config.bias_gyro,
            self.config.drift_rate_gyro,
            t,
        )

        return {
            "time": t,
            "body_acc": body_acc,
            "total_acc": total_acc,
            "gyro": gyro,
            "metadata": {
                "is_synthetic": True,
                "data_source": "SYNTHETIC_IMU_GENERATOR",
                "activity": activity,
                "cadence_hz": freq,
                "duration_sec": t[-1] + (t[1] - t[0]),
                "sampling_rate_hz": self.config.sampling_rate_hz,
            },
        }

    def generate_window(
        self,
        activity: str = "WALKING",
        duration_sec: Optional[float] = None,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """Generate a single window for any supported activity."""
        if activity not in SUPPORTED_ACTIVITIES:
            raise ValueError(f"Unsupported activity '{activity}'. Supported: {SUPPORTED_ACTIVITIES}")

        if activity in ["SITTING", "STANDING", "LAYING"]:
            return self.generate_stationary_window(activity, duration_sec=duration_sec)
        return self.generate_dynamic_window(activity, duration_sec=duration_sec)

    def generate_batch_dataset(
        self,
        samples_per_activity: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate a synthetic multi-activity test batch formatted like UCI HAR tensors.

        Output:
            X: Tensor of shape (num_windows, 128, 9)
            y: 1D array of activity labels (0-indexed: 0..5)
            metadata: Verification dictionary with SYNTHETIC markers
        """
        windows = []
        labels = []
        activity_to_id = {act: i for i, act in enumerate(SUPPORTED_ACTIVITIES)}

        for act in SUPPORTED_ACTIVITIES:
            for _ in range(samples_per_activity):
                w = self.generate_window(act)
                # Concatenate: [body_acc (3), gyro (3), total_acc (3)] -> 9 channels
                tensor_9ch = np.concatenate([w["body_acc"], w["gyro"], w["total_acc"]], axis=-1)
                windows.append(tensor_9ch)
                labels.append(activity_to_id[act])

        X = np.stack(windows, axis=0)  # Shape (N, 128, 9)
        y = np.array(labels, dtype=np.int64)

        metadata = {
            "is_synthetic": True,
            "WARNING": "SYNTHETIC SENSOR DATA - DO NOT COMBINE WITH EMPIRICAL DATA FOR EVALUATION",
            "samples": len(X),
            "timesteps": X.shape[1],
            "channels": X.shape[2],
            "channel_order": [
                "body_acc_x", "body_acc_y", "body_acc_z",
                "body_gyro_x", "body_gyro_y", "body_gyro_z",
                "total_acc_x", "total_acc_y", "total_acc_z",
            ],
            "activity_map": {str(v): k for k, v in activity_to_id.items()},
        }
        return X, y, metadata


if __name__ == "__main__":
    generator = SyntheticIMUGenerator()
    X_syn, y_syn, meta = generator.generate_batch_dataset(samples_per_activity=10)
    print(f"[SYNTHETIC GENERATOR DEMO] Generated tensor shape: {X_syn.shape}")
    print(f"[SYNTHETIC GENERATOR DEMO] Labels distribution: {np.bincount(y_syn)}")
    print(f"[SYNTHETIC GENERATOR DEMO] Synthetic verification flag: {meta['is_synthetic']}")
