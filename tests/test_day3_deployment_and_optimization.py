"""Unit and integration test suite for Day 3 model optimization and streaming deployment.

Verifies:
1. Baseline Day 2 model artifact integrity and immutable SHA256.
2. ONNX export, structural checker, and runtime session creation.
3. Numerical parity between PyTorch and ONNX Runtime.
4. Dynamic batch size handling.
5. FP16 conversion and graph topological sort integrity.
6. INT8 quantization (dynamic and static QDQ) and graph operator inspection.
7. Calibration data reader format and test-set isolation.
8. Sliding-window buffer channel validation, NaN/Inf rejection, and 50% overlap.
9. Real-time stream simulator prediction output format and latency collection.
10. Benchmark latency profiling and metrics calculation.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from src.data.preprocess import ChannelWiseStandardScaler
from src.deployment.benchmark import benchmark_latency, evaluate_classifier_predictions
from src.deployment.onnx_export import export_pytorch_to_onnx, verify_numerical_parity
from src.deployment.quantization import (
    HARCalibrationDataReader,
    convert_to_fp16,
    inspect_onnx_graph,
    quantize_dynamic_int8,
    quantize_static_int8,
)
from src.deployment.streaming import RealTimeStreamSimulator, SlidingWindowBuffer
from src.models.cnn_1d import CNN1DHAR, Conv1DNet

DAY2_BASELINE_SHA256 = "19c5ba643e17f2cada7a446c536b154a00ad7affaadda43fae67e8f7c96a55ba"
DAY2_BASELINE_SIZE = 420729


def test_day2_baseline_model_unmodified_sha256() -> None:
    """Requirement 1: Verify baseline Day 2 model artifact remains completely untouched."""
    repo_root = Path(__file__).resolve().parent.parent
    baseline_pt = repo_root / "models" / "cnn_1d.pt"
    assert baseline_pt.exists(), "Day 2 baseline model checkpoint missing!"

    data = baseline_pt.read_bytes()
    assert len(data) == DAY2_BASELINE_SIZE, f"Baseline size changed: {len(data)} vs {DAY2_BASELINE_SIZE}"
    current_sha = hashlib.sha256(data).hexdigest()
    assert current_sha == DAY2_BASELINE_SHA256, (
        f"Baseline Day 2 model was modified! Expected {DAY2_BASELINE_SHA256}, got {current_sha}"
    )


def test_onnx_fp32_export_and_structural_validity(tmp_path: Path) -> None:
    """Requirements 2 & 3: Export PyTorch CNN to ONNX and verify graph validity."""
    repo_root = Path(__file__).resolve().parent.parent
    baseline_pt = repo_root / "models" / "cnn_1d.pt"
    out_onnx = tmp_path / "test_cnn.onnx"

    exported_path = export_pytorch_to_onnx(
        model_path=baseline_pt,
        output_path=out_onnx,
        opset_version=17,
        dynamic_batch=True,
        embed_weights=True,
    )
    assert exported_path.exists()
    assert exported_path.stat().st_size > 0

    # Validate with onnx.checker
    model = onnx.load(str(exported_path))
    onnx.checker.check_model(model)

    # Validate ONNX Runtime session loads without errors
    session = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "sensor_window"
    assert session.get_outputs()[0].name == "activity_logits"


def test_pytorch_onnx_numerical_parity(tmp_path: Path) -> None:
    """Requirement 4: Verify numerical parity between PyTorch and ONNX outputs."""
    repo_root = Path(__file__).resolve().parent.parent
    baseline_pt = repo_root / "models" / "cnn_1d.pt"
    out_onnx = tmp_path / "test_parity.onnx"

    export_pytorch_to_onnx(model_path=baseline_pt, output_path=out_onnx)

    rng = np.random.RandomState(42)
    # Generate 20 synthetic windows: shape (20, 128, 9)
    synthetic_samples = rng.randn(20, 128, 9).astype(np.float32)

    results = verify_numerical_parity(
        model_pt_path=baseline_pt,
        model_onnx_path=out_onnx,
        test_samples=synthetic_samples,
        atol=1e-4,
    )

    assert results["parity_passed"] is True
    assert results["max_absolute_logit_diff"] < 1e-4
    assert results["prediction_agreement_pct"] == 100.0


def test_dynamic_batch_size_support(tmp_path: Path) -> None:
    """Requirement 5: Verify exported ONNX model supports dynamic batch sizes (1, 4, 16)."""
    repo_root = Path(__file__).resolve().parent.parent
    baseline_pt = repo_root / "models" / "cnn_1d.pt"
    out_onnx = tmp_path / "test_dynamic_batch.onnx"

    export_pytorch_to_onnx(model_path=baseline_pt, output_path=out_onnx, dynamic_batch=True)
    session = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])

    for b_size in [1, 4, 16]:
        dummy = np.random.randn(b_size, 9, 128).astype(np.float32)
        out = session.run(["activity_logits"], {"sensor_window": dummy})[0]
        assert out.shape == (b_size, 6), f"Expected shape ({b_size}, 6), got {out.shape}"


def test_fp16_conversion_and_graph_integrity(tmp_path: Path) -> None:
    """Requirement 6: Verify FP16 conversion, topological ordering, and file size reduction."""
    repo_root = Path(__file__).resolve().parent.parent
    baseline_pt = repo_root / "models" / "cnn_1d.pt"
    fp32_onnx = tmp_path / "fp32.onnx"
    fp16_onnx = tmp_path / "fp16.onnx"

    export_pytorch_to_onnx(model_path=baseline_pt, output_path=fp32_onnx)
    convert_to_fp16(input_onnx_path=fp32_onnx, output_onnx_path=fp16_onnx)

    assert fp16_onnx.exists()
    assert fp16_onnx.stat().st_size < fp32_onnx.stat().st_size

    # Validate ONNX checker and session loading
    model_proto = onnx.load(str(fp16_onnx))
    onnx.checker.check_model(model_proto)

    session = ort.InferenceSession(str(fp16_onnx), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 9, 128).astype(np.float32)
    out = session.run(None, {"sensor_window": dummy})[0]
    assert out.shape == (1, 6)


def test_int8_quantization_and_graph_inspection(tmp_path: Path) -> None:
    """Requirement 7: Verify INT8 quantization and graph operator inspection."""
    repo_root = Path(__file__).resolve().parent.parent
    baseline_pt = repo_root / "models" / "cnn_1d.pt"
    fp32_onnx = tmp_path / "fp32.onnx"
    int8_dyn_onnx = tmp_path / "int8_dyn.onnx"

    export_pytorch_to_onnx(model_path=baseline_pt, output_path=fp32_onnx)
    quantize_dynamic_int8(input_onnx_path=fp32_onnx, output_onnx_path=int8_dyn_onnx)

    assert int8_dyn_onnx.exists()
    assert int8_dyn_onnx.stat().st_size < fp32_onnx.stat().st_size

    info = inspect_onnx_graph(int8_dyn_onnx)
    assert info["precision_type"] == "int8_dynamic"
    assert info["conv_layers_quantized"] == 3
    assert info["linear_layers_quantized"] == 2
    assert "ConvInteger" in info["operator_distribution"]
    assert "MatMulInteger" in info["operator_distribution"]


def test_calibration_data_reader_format_and_rewind() -> None:
    """Requirement 8: Test HARCalibrationDataReader format, iteration, and rewind."""
    rng = np.random.RandomState(42)
    dummy_windows = rng.randn(10, 128, 9).astype(np.float32)

    reader = HARCalibrationDataReader(dummy_windows, input_name="sensor_window")
    batch = reader.get_next()
    assert batch is not None
    assert "sensor_window" in batch
    assert batch["sensor_window"].shape == (1, 9, 128)

    # Consume all remaining
    count = 1
    while reader.get_next() is not None:
        count += 1
    assert count == 10

    # Rewind and verify restarts
    reader.rewind()
    batch_again = reader.get_next()
    assert batch_again is not None
    assert np.allclose(batch["sensor_window"], batch_again["sensor_window"])


def test_static_int8_calibration_with_reader(tmp_path: Path) -> None:
    """Requirement 9: Test static INT8 quantization with calibration data reader."""
    repo_root = Path(__file__).resolve().parent.parent
    baseline_pt = repo_root / "models" / "cnn_1d.pt"
    fp32_onnx = tmp_path / "fp32.onnx"
    static_int8_onnx = tmp_path / "static_int8.onnx"

    export_pytorch_to_onnx(model_path=baseline_pt, output_path=fp32_onnx)

    rng = np.random.RandomState(42)
    calib_data = rng.randn(20, 128, 9).astype(np.float32)
    quantize_static_int8(fp32_onnx, static_int8_onnx, calibration_data=calib_data)

    assert static_int8_onnx.exists()
    info = inspect_onnx_graph(static_int8_onnx)
    assert info["precision_type"] == "int8_static_qdq"
    assert "QuantizeLinear" in info["operator_distribution"]
    assert "DequantizeLinear" in info["operator_distribution"]


def test_sliding_window_buffer_initialization() -> None:
    """Requirement 10: Test SlidingWindowBuffer initialization properties."""
    buf = SlidingWindowBuffer(window_size=128, stride=64, channels=9, sampling_rate_hz=50.0)
    assert buf.window_size == 128
    assert buf.stride == 64
    assert buf.channels == 9
    assert buf.buffer_length == 0
    assert buf.windows_emitted == 0
    assert buf.is_window_ready() is False

    with pytest.raises(ValueError):
        SlidingWindowBuffer(window_size=0)
    with pytest.raises(ValueError):
        SlidingWindowBuffer(window_size=128, stride=129)  # stride > window_size


def test_sliding_window_buffer_9_channel_validation() -> None:
    """Requirement 11: Assert SlidingWindowBuffer rejects incorrect channel counts."""
    buf = SlidingWindowBuffer(window_size=128, stride=64, channels=9)

    # Correct shape
    buf.add_sample(np.zeros(9))
    assert buf.buffer_length == 1

    # Invalid channel count (8 channels)
    with pytest.raises(ValueError, match="Channel dimension mismatch"):
        buf.add_sample(np.zeros(8))

    # Invalid channel count in batch (12 channels)
    with pytest.raises(ValueError, match="Channel dimension mismatch"):
        buf.add_samples(np.zeros((10, 12)))


def test_sliding_window_buffer_nan_inf_rejection() -> None:
    """Requirement 12: Assert SlidingWindowBuffer rejects NaN and Infinite values."""
    buf = SlidingWindowBuffer(window_size=128, stride=64, channels=9)

    sample_nan = np.zeros(9)
    sample_nan[2] = np.nan
    with pytest.raises(ValueError, match="NaN or Infinite"):
        buf.add_sample(sample_nan)

    sample_inf = np.zeros(9)
    sample_inf[4] = np.inf
    with pytest.raises(ValueError, match="NaN or Infinite"):
        buf.add_sample(sample_inf)


def test_sliding_window_buffer_50_percent_overlap() -> None:
    """Requirements 13-15: Test window accumulation, 50% overlap, and stride advancement."""
    buf = SlidingWindowBuffer(window_size=128, stride=64, channels=9)

    # Feed 128 samples: sequence 0..127
    samples_1 = np.arange(128 * 9, dtype=np.float32).reshape(128, 9)
    buf.add_samples(samples_1)
    assert buf.is_window_ready() is True

    window_1 = buf.get_window()
    assert window_1.shape == (128, 9)
    assert np.array_equal(window_1, samples_1)

    # Advance by stride (64)
    buf.advance()
    assert buf.buffer_length == 64
    assert buf.windows_emitted == 1
    assert buf.is_window_ready() is False

    # Feed another 64 samples: sequence 128..191
    samples_2 = np.arange(128 * 9, 192 * 9, dtype=np.float32).reshape(64, 9)
    buf.add_samples(samples_2)
    assert buf.is_window_ready() is True

    window_2 = buf.get_window()
    assert window_2.shape == (128, 9)

    # 50% overlap assertion: last 64 of window_1 MUST equal first 64 of window_2
    assert np.array_equal(window_1[64:], window_2[:64])
    # The new 64 samples of window_2 MUST equal samples_2
    assert np.array_equal(window_2[64:], samples_2)


def test_real_time_stream_simulator_prediction_format() -> None:
    """Requirements 16 & 17: Test RealTimeStreamSimulator prediction output format."""
    repo_root = Path(__file__).resolve().parent.parent
    onnx_path = repo_root / "models" / "optimized" / "cnn_1d_fp32.onnx"

    scaler = ChannelWiseStandardScaler()
    dummy_train = np.ones((5, 128, 9), dtype=np.float32)
    scaler.fit(dummy_train)

    sim = RealTimeStreamSimulator(
        scaler=scaler,
        backend_type="onnx",
        model_path=onnx_path,
        sampling_rate_hz=50.0,
        window_size=128,
        stride=64,
    )

    dummy_win = np.random.randn(128, 9).astype(np.float32)
    pred_idx, class_name, confidence, lat_ms = sim.predict_window(dummy_win)

    assert isinstance(pred_idx, int)
    assert 0 <= pred_idx < 6
    assert isinstance(class_name, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
    assert isinstance(lat_ms, float)
    assert lat_ms > 0.0


def test_real_time_stream_simulator_deadline_and_streaming() -> None:
    """Requirements 18 & 19: Test end-to-end stream processing and deadline metrics."""
    repo_root = Path(__file__).resolve().parent.parent
    onnx_path = repo_root / "models" / "optimized" / "cnn_1d_fp32.onnx"

    scaler = ChannelWiseStandardScaler()
    dummy_train = np.ones((5, 128, 9), dtype=np.float32)
    scaler.fit(dummy_train)

    sim = RealTimeStreamSimulator(
        scaler=scaler,
        backend_type="onnx",
        model_path=onnx_path,
        sampling_rate_hz=50.0,
        window_size=128,
        stride=64,
    )

    # 256 samples stream = 2 windows emitted (at 128, then at 192)
    stream_data = np.random.randn(256, 9).astype(np.float32)
    results = sim.process_stream(stream_data, chunk_size=32)

    assert results["total_samples_processed"] == 256
    assert results["total_windows_emitted"] >= 2
    assert results["total_predictions"] >= 2
    assert results["nominal_deadline_ms"] == 1280.0
    assert results["missed_deadlines"] == 0
    assert results["mean_inference_latency_ms"] > 0.0


def test_benchmark_latency_profiling() -> None:
    """Requirement 20: Test benchmark_latency statistics calculation."""
    def dummy_runner(x: np.ndarray) -> np.ndarray:
        return x * 2.0

    dummy_input = np.ones((1, 9, 128), dtype=np.float32)
    stats = benchmark_latency(dummy_runner, dummy_input, warm_up=10, repetitions=50)

    assert stats["warm_up_iterations"] == 10
    assert stats["timed_iterations"] == 50
    assert stats["mean_latency_ms"] >= 0.0
    assert stats["median_latency_ms"] >= 0.0
    assert stats["p95_latency_ms"] >= stats["median_latency_ms"]
    assert stats["throughput_windows_per_sec"] > 0.0


def test_evaluate_classifier_metrics() -> None:
    """Test evaluate_classifier_predictions accuracy, precision, recall, and F1 calculations."""
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([0, 1, 2, 3, 4, 5])

    metrics = evaluate_classifier_predictions(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["accuracy_pct"] == 100.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["f1_weighted"] == 1.0
    assert len(metrics["per_class_f1"]) == 6
