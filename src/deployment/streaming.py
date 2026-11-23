"""Real-time sliding-window IMU streaming simulation and telemetry processing.

Simulates edge streaming telemetry across continuous multi-activity timelines,
buffering triaxial inertial sensor channels with 50% overlap, standardizing windows
via the training-fitted scaler, and executing optimized low-latency inference.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch

from src.data.preprocess import ChannelWiseStandardScaler
from src.models.cnn_1d import ACTIVITY_NAMES, CNN1DHAR

logger = logging.getLogger(__name__)


class SlidingWindowBuffer:
    """Sliding FIFO ring buffer for multi-channel streaming sensor signals.

    Accumulates continuous 9-channel IMU samples at a nominal 50 Hz rate, triggering
    a window-ready event when 128 samples (2.56s) are available and advancing with
    a 64-sample stride (50% overlap, 1.28s step size).
    """

    def __init__(
        self,
        window_size: int = 128,
        stride: int = 64,
        channels: int = 9,
        sampling_rate_hz: float = 50.0,
    ):
        """Initialize buffer parameters.

        Args:
            window_size: Number of timesteps per window (default: 128).
            stride: Step size between successive windows (default: 64).
            channels: Expected sensor channel count (default: 9).
            sampling_rate_hz: Nominal sensor sampling frequency in Hz (default: 50.0).
        """
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive integers.")
        if stride > window_size:
            raise ValueError("stride cannot exceed window_size.")
        if channels <= 0:
            raise ValueError("channels must be a positive integer.")

        self.window_size = window_size
        self.stride = stride
        self.channels = channels
        self.sampling_rate_hz = sampling_rate_hz

        # Internal buffer shape: (N_buffered, channels)
        self._buffer: np.ndarray = np.empty((0, channels), dtype=np.float32)
        self._total_samples_received: int = 0
        self._windows_emitted: int = 0

    def add_sample(self, sample: np.ndarray) -> None:
        """Add a single 9-channel sensor reading at timestep t.

        Args:
            sample: 1D array of shape (9,) or 2D array of shape (1, 9).
        """
        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim == 1:
            sample = np.expand_dims(sample, axis=0)

        self._validate_input(sample)
        self._buffer = np.vstack([self._buffer, sample])
        self._total_samples_received += len(sample)

    def add_samples(self, samples: np.ndarray) -> None:
        """Add a sequence of continuous multi-channel sensor readings.

        Args:
            samples: 2D array of shape (T, 9).
        """
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = np.expand_dims(samples, axis=0)

        self._validate_input(samples)
        self._buffer = np.vstack([self._buffer, samples])
        self._total_samples_received += len(samples)

    def _validate_input(self, data: np.ndarray) -> None:
        """Assert channel dimensionality and absence of numerical corruptions."""
        if data.shape[1] != self.channels:
            raise ValueError(
                f"Channel dimension mismatch: expected {self.channels} channels, got {data.shape[1]}."
            )
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("Input sensor data contains NaN or Infinite values.")

    def is_window_ready(self) -> bool:
        """Return True when at least `window_size` samples are buffered."""
        return len(self._buffer) >= self.window_size

    def get_window(self) -> np.ndarray:
        """Return the current window of shape (128, 9).

        Raises:
            RuntimeError: If fewer than `window_size` samples are currently buffered.
        """
        if not self.is_window_ready():
            raise RuntimeError(
                f"Insufficient samples in buffer: {len(self._buffer)}/{self.window_size}."
            )
        return self._buffer[: self.window_size].copy()

    def advance(self) -> None:
        """Advance the buffer forward by `stride` (64 samples).

        Raises:
            RuntimeError: If the buffer contains fewer than `stride` samples.
        """
        if len(self._buffer) < self.stride:
            raise RuntimeError("Cannot advance buffer: fewer samples than stride.")
        self._buffer = self._buffer[self.stride :]
        self._windows_emitted += 1

    def reset(self) -> None:
        """Reset internal buffer state."""
        self._buffer = np.empty((0, self.channels), dtype=np.float32)
        self._total_samples_received = 0
        self._windows_emitted = 0

    @property
    def buffer_length(self) -> int:
        """Current number of samples buffered."""
        return len(self._buffer)

    @property
    def total_samples_received(self) -> int:
        """Cumulative count of received samples."""
        return self._total_samples_received

    @property
    def windows_emitted(self) -> int:
        """Count of windows extracted and advanced."""
        return self._windows_emitted


class RealTimeStreamSimulator:
    """Simulates real-time edge streaming inference over continuous IMU telemetry.

    Orchestrates:
    1. SlidingWindowBuffer (128 samples, 50% overlap at 50 Hz).
    2. ChannelWiseStandardScaler (zero-leakage training-fitted parameters).
    3. Low-latency inference backend (PyTorch or ONNX Runtime).
    4. Per-window performance and deadline monitoring.
    """

    def __init__(
        self,
        scaler: ChannelWiseStandardScaler,
        backend_type: str = "onnx",
        model_path: Optional[Union[Path, str]] = None,
        sampling_rate_hz: float = 50.0,
        window_size: int = 128,
        stride: int = 64,
    ):
        """Initialize the streaming simulator.

        Args:
            scaler: Fitted ChannelWiseStandardScaler for sensor channel normalization.
            backend_type: Inference backend ('onnx' or 'pytorch').
            model_path: Path to .onnx model or .pt checkpoint.
            sampling_rate_hz: Nominal sensor sampling rate in Hz (default: 50.0).
            window_size: Window size in timesteps (default: 128).
            stride: Stride in timesteps (default: 64).
        """
        self.scaler = scaler
        self.backend_type = backend_type.lower()
        self.sampling_rate_hz = sampling_rate_hz
        self.window_size = window_size
        self.stride = stride

        # Nominal budget between consecutive windows (stride / sampling_rate)
        # e.g. 64 / 50 Hz = 1.28 seconds (1280 ms)
        self.nominal_deadline_ms = (self.stride / self.sampling_rate_hz) * 1000.0

        self.buffer = SlidingWindowBuffer(
            window_size=self.window_size,
            stride=self.stride,
            channels=9,
            sampling_rate_hz=self.sampling_rate_hz,
        )

        self._init_backend(model_path)

    def _init_backend(self, model_path: Optional[Union[Path, str]]) -> None:
        """Initialize the selected inference runtime engine."""
        if self.backend_type == "onnx":
            if model_path is None:
                model_path = Path("models/optimized/cnn_1d_fp32.onnx")
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        elif self.backend_type == "pytorch":
            if model_path is None:
                model_path = Path("models/cnn_1d.pt")
            har = CNN1DHAR.load(model_path, device="cpu")
            self.pytorch_model = har.model
            self.pytorch_model.eval()
        else:
            raise ValueError(f"Unsupported backend_type: {self.backend_type}. Choose 'onnx' or 'pytorch'.")

    def predict_window(self, window_128x9: np.ndarray) -> Tuple[int, str, float, float]:
        """Normalize and infer activity on a single 128x9 window.

        Args:
            window_128x9: Raw sensor window of shape (128, 9).

        Returns:
            Tuple[int, str, float, float]: (class_idx, class_name, confidence, latency_ms)
        """
        # 1. Standardize using training scaler
        scaled_window = self.scaler.transform(np.expand_dims(window_128x9, axis=0))[0]  # (128, 9)

        # 2. Format tensor: transpose to (1, 9, 128)
        tensor_in = np.transpose(np.expand_dims(scaled_window, axis=0), (0, 2, 1)).astype(np.float32)

        t0 = time.perf_counter()
        if self.backend_type == "onnx":
            logits = self.session.run([self.output_name], {self.input_name: tensor_in})[0][0]
        else:
            with torch.no_grad():
                logits = self.pytorch_model(torch.from_numpy(tensor_in)).cpu().numpy()[0]
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0

        # Compute softmax probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        class_name = ACTIVITY_NAMES[pred_idx] if pred_idx < len(ACTIVITY_NAMES) else f"CLASS_{pred_idx}"

        return pred_idx, class_name, confidence, latency_ms

    def process_stream(
        self,
        stream_data: np.ndarray,
        chunk_size: int = 16,
    ) -> Dict[str, Any]:
        """Simulate real-time streaming over a multi-channel sensor recording.

        Args:
            stream_data: Array of shape (T, 9) representing continuous IMU readings.
            chunk_size: Simulated telemetry arrival batch size (e.g. 16 samples = 320ms at 50 Hz).

        Returns:
            Dict[str, Any]: Telemetry stream processing log and aggregated metrics.
        """
        self.buffer.reset()
        predictions_log: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []
        start_wall_time = time.perf_counter()

        n_samples = len(stream_data)
        window_counter = 0

        for offset in range(0, n_samples, chunk_size):
            chunk = stream_data[offset : offset + chunk_size]
            self.buffer.add_samples(chunk)

            while self.buffer.is_window_ready():
                window = self.buffer.get_window()
                pred_idx, class_name, confidence, lat_ms = self.predict_window(window)
                latencies_ms.append(lat_ms)

                timestamp_s = (window_counter * self.stride) / self.sampling_rate_hz
                deadline_missed = bool(lat_ms > self.nominal_deadline_ms)

                predictions_log.append({
                    "window_index": window_counter,
                    "timestamp_sec": round(timestamp_s, 2),
                    "predicted_class": pred_idx,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "latency_ms": round(lat_ms, 4),
                    "deadline_missed": deadline_missed,
                })

                window_counter += 1
                self.buffer.advance()

        total_wall_sec = time.perf_counter() - start_wall_time
        num_preds = len(latencies_ms)

        if num_preds > 0:
            mean_lat = float(np.mean(latencies_ms))
            median_lat = float(np.median(latencies_ms))
            p95_lat = float(np.percentile(latencies_ms, 95))
            max_lat = float(np.max(latencies_ms))
            throughput = float(num_preds / total_wall_sec)
            missed_count = sum(1 for p in predictions_log if p["deadline_missed"])
        else:
            mean_lat = median_lat = p95_lat = max_lat = throughput = 0.0
            missed_count = 0

        return {
            "backend": self.backend_type,
            "total_samples_processed": n_samples,
            "simulated_stream_duration_sec": round(n_samples / self.sampling_rate_hz, 2),
            "total_windows_emitted": window_counter,
            "total_predictions": num_preds,
            "mean_inference_latency_ms": round(mean_lat, 4),
            "median_inference_latency_ms": round(median_lat, 4),
            "p95_inference_latency_ms": round(p95_lat, 4),
            "max_inference_latency_ms": round(max_lat, 4),
            "nominal_deadline_ms": round(self.nominal_deadline_ms, 2),
            "missed_deadlines": missed_count,
            "windows_per_second": round(throughput, 2),
            "total_wall_time_sec": round(total_wall_sec, 4),
            "predictions_sample": predictions_log[:10],
        }
