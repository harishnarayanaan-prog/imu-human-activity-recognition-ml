"""Deployment, model optimization, and edge streaming emulation package.

Provides modules for:
- PyTorch to ONNX export with numerical parity verification
- FP16 and INT8 (static QDQ and dynamic) quantization
- Benchmarking of optimized models (latency, throughput, footprint, classification metrics)
- Real-time sliding-window IMU streaming emulation
"""

from src.deployment.onnx_export import export_pytorch_to_onnx, verify_numerical_parity
from src.deployment.quantization import (
    convert_to_fp16,
    quantize_static_int8,
    quantize_dynamic_int8,
    inspect_onnx_graph,
    HARCalibrationDataReader,
)
from src.deployment.benchmark import benchmark_latency, evaluate_classifier
from src.deployment.streaming import SlidingWindowBuffer, RealTimeStreamSimulator

__all__ = [
    "export_pytorch_to_onnx",
    "verify_numerical_parity",
    "convert_to_fp16",
    "quantize_static_int8",
    "quantize_dynamic_int8",
    "inspect_onnx_graph",
    "HARCalibrationDataReader",
    "benchmark_latency",
    "evaluate_classifier",
    "SlidingWindowBuffer",
    "RealTimeStreamSimulator",
]
