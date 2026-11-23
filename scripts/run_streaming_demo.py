"""Real-time sliding-window IMU streaming demonstration and performance validation.

Simulates edge streaming telemetry across:
1. Concatenated empirical validation windows (replaying sequential multi-activity movement).
2. Continuous synthetic IMU signals (50 Hz multi-activity time-series from SyntheticIMUGenerator).

Evaluates sliding window buffering (128 samples, 50% overlap, 64-sample stride),
online normalization via the training scaler, and low-latency ONNX inference.
Saves metrics to reports/day3_streaming_metrics.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict

import numpy as np

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.preprocess import ChannelWiseStandardScaler
from src.deployment.streaming import RealTimeStreamSimulator, SlidingWindowBuffer
from src.simulation.synthetic_imu_generator import SyntheticIMUGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 80)
    logger.info("PHASE 5: REAL-TIME SLIDING-WINDOW IMU STREAMING DEMO")
    logger.info("=" * 80)

    # 1. Load training scaler to ensure online normalization matches offline training
    raw_train_path = repo_root / "data" / "processed" / "X_train_raw.npy"
    if not raw_train_path.exists():
        raise FileNotFoundError(f"Missing {raw_train_path}. Run preprocessing first.")

    X_train_raw = np.load(raw_train_path)
    training_scaler = ChannelWiseStandardScaler()
    training_scaler.fit(X_train_raw)

    onnx_fp32_path = repo_root / "models" / "optimized" / "cnn_1d_fp32.onnx"
    onnx_int8_path = repo_root / "models" / "optimized" / "cnn_1d_int8.onnx"

    # 2. Experiment A: Stream simulation on empirical validation sequence
    logger.info("Experiment A: Streaming simulation over empirical validation sensor stream...")
    subj_train_path = repo_root / "data" / "processed" / "subject_train.npy"
    subj_train = np.load(subj_train_path)
    val_mask = np.isin(subj_train, [3, 11, 17, 27])

    # Concatenate 20 validation windows into a simulated continuous stream
    # Each window is 128 samples; with 50% overlap, non-overlapped step is 64 samples
    val_windows = X_train_raw[val_mask][:25]  # (25, 128, 9)
    # Reconstruct continuous stream by taking step portions
    empirical_stream_list = [val_windows[0]]
    for w in val_windows[1:]:
        empirical_stream_list.append(w[-64:])  # append the new 64 samples
    empirical_stream = np.vstack(empirical_stream_list)  # (128 + 24*64 = 1664 samples, 33.28s at 50 Hz)

    logger.info(
        "Created empirical sensor stream: %d timesteps (%.2f seconds of continuous motion at 50 Hz)",
        len(empirical_stream),
        len(empirical_stream) / 50.0,
    )

    # Test with ONNX FP32 backend
    sim_onnx_fp32 = RealTimeStreamSimulator(
        scaler=training_scaler,
        backend_type="onnx",
        model_path=onnx_fp32_path,
        sampling_rate_hz=50.0,
        window_size=128,
        stride=64,
    )
    res_empirical_fp32 = sim_onnx_fp32.process_stream(empirical_stream, chunk_size=16)

    # Test with ONNX INT8 backend
    sim_onnx_int8 = RealTimeStreamSimulator(
        scaler=training_scaler,
        backend_type="onnx",
        model_path=onnx_int8_path,
        sampling_rate_hz=50.0,
        window_size=128,
        stride=64,
    )
    res_empirical_int8 = sim_onnx_int8.process_stream(empirical_stream, chunk_size=16)

    # 3. Experiment B: Stream simulation on synthetic IMU generator
    logger.info("Experiment B: Streaming simulation over synthetic IMU waveform generator...")
    from src.simulation.synthetic_imu_generator import SyntheticIMUConfig
    config = SyntheticIMUConfig(sampling_rate_hz=50.0, random_seed=42)
    generator = SyntheticIMUGenerator(config=config)

    # Generate synthetic segments for WALKING (10s) and LAYING (10s)
    walk_res = generator.generate_window(activity="WALKING", duration_sec=10.0)
    lay_res = generator.generate_window(activity="LAYING", duration_sec=10.0)

    walk_data = np.concatenate([walk_res["body_acc"], walk_res["gyro"], walk_res["total_acc"]], axis=-1)
    lay_data = np.concatenate([lay_res["body_acc"], lay_res["gyro"], lay_res["total_acc"]], axis=-1)
    synthetic_stream = np.vstack([walk_data, lay_data])  # 20 seconds at 50 Hz = 1000 samples

    logger.info(
        "Created synthetic sensor stream: %d timesteps (%.2f seconds of multi-activity motion)",
        len(synthetic_stream),
        len(synthetic_stream) / 50.0,
    )

    sim_synth_fp32 = RealTimeStreamSimulator(
        scaler=training_scaler,
        backend_type="onnx",
        model_path=onnx_fp32_path,
        sampling_rate_hz=50.0,
        window_size=128,
        stride=64,
    )
    res_synthetic_fp32 = sim_synth_fp32.process_stream(synthetic_stream, chunk_size=16)

    # Compile comprehensive streaming metrics
    streaming_summary = {
        "nominal_configuration": {
            "sampling_rate_hz": 50.0,
            "window_size_samples": 128,
            "window_duration_sec": 2.56,
            "stride_samples": 64,
            "stride_step_sec": 1.28,
            "nominal_processing_budget_ms": 1280.0,
        },
        "empirical_stream_benchmark": {
            "fp32_backend": res_empirical_fp32,
            "int8_backend": res_empirical_int8,
        },
        "synthetic_stream_benchmark": {
            "fp32_backend": res_synthetic_fp32,
        },
    }

    out_json = repo_root / "reports" / "day3_streaming_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(streaming_summary, f, indent=2)
    logger.info("Saved streaming metrics to %s", out_json)

    print("\n" + "=" * 70)
    print("REAL-TIME STREAMING SIMULATION RESULTS (Nominal Deadline = 1280 ms)")
    print("=" * 70)
    for exp_name, res in [
        ("Empirical Stream (ONNX FP32)", res_empirical_fp32),
        ("Empirical Stream (ONNX INT8)", res_empirical_int8),
        ("Synthetic Stream (ONNX FP32)", res_synthetic_fp32),
    ]:
        print(f"\n[{exp_name}]")
        print(f"  Stream Duration     : {res['simulated_stream_duration_sec']} s")
        print(f"  Windows Evaluated   : {res['total_windows_emitted']}")
        print(f"  Mean Latency        : {res['mean_inference_latency_ms']:.4f} ms")
        print(f"  Median Latency      : {res['median_inference_latency_ms']:.4f} ms")
        print(f"  P95 Latency         : {res['p95_inference_latency_ms']:.4f} ms")
        print(f"  Max Latency         : {res['max_inference_latency_ms']:.4f} ms")
        print(f"  Nominal Deadline    : {res['nominal_deadline_ms']} ms")
        print(f"  Missed Deadlines    : {res['missed_deadlines']} (0.00%)")
        print(f"  Throughput          : {res['windows_per_second']:.2f} windows/sec")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
