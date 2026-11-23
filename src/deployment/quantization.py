"""Model precision optimization and quantization pipelines.

Supports:
- FP16 conversion via onnxruntime.transformers.float16
- Static INT8 quantization using internal validation calibration data (QDQ format)
- Dynamic INT8 quantization for weight-only / dynamic activation quantization
- In-depth ONNX computational graph operator inspection
"""

from __future__ import annotations

import logging
from collections import Counter
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.transformers.float16 import convert_float_to_float16

logger = logging.getLogger(__name__)


class HARCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader feeding internal validation/training windows to ONNX Runtime.

    Guarantees strict isolation: accepts only internal data; official test data is never used.
    """

    def __init__(self, calibration_windows: np.ndarray, input_name: str = "sensor_window"):
        """Initialize calibration reader.

        Args:
            calibration_windows: Array of shape (N, 128, 9) or (N, 9, 128).
            input_name: ONNX model input tensor name (default: "sensor_window").
        """
        super().__init__()
        self.input_name = input_name

        if calibration_windows.ndim == 2:
            calibration_windows = np.expand_dims(calibration_windows, axis=0)

        if calibration_windows.shape[1] == 128 and calibration_windows.shape[2] == 9:
            self.data = np.transpose(calibration_windows, (0, 2, 1)).astype(np.float32)
        elif calibration_windows.shape[1] == 9 and calibration_windows.shape[2] == 128:
            self.data = calibration_windows.astype(np.float32)
        else:
            raise ValueError(
                f"Invalid calibration data shape: {calibration_windows.shape}. "
                "Expected (N, 128, 9) or (N, 9, 128)."
            )

        self.enum_data = iter([{self.input_name: np.expand_dims(x, axis=0)} for x in self.data])

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the next calibration input dictionary or None when exhausted."""
        return next(self.enum_data, None)

    def rewind(self) -> None:
        """Rewind the data iterator for multiple calibration passes."""
        self.enum_data = iter([{self.input_name: np.expand_dims(x, axis=0)} for x in self.data])


def _topological_sort_graph(model: onnx.ModelProto) -> onnx.ModelProto:
    """Ensure all nodes in graph are in valid topological order."""
    graph = model.graph
    known_outputs = {inp.name for inp in graph.input} | {init.name for init in graph.initializer}
    sorted_nodes = []
    remaining_nodes = list(graph.node)

    while remaining_nodes:
        progress = False
        for i, node in enumerate(remaining_nodes):
            if all(inp in known_outputs or inp == "" for inp in node.input):
                sorted_nodes.append(node)
                for out in node.output:
                    known_outputs.add(out)
                remaining_nodes.pop(i)
                progress = True
                break
        if not progress:
            # Fallback if circular dependency or unresolvable node
            sorted_nodes.extend(remaining_nodes)
            break

    del graph.node[:]
    graph.node.extend(sorted_nodes)
    return model


def convert_to_fp16(
    input_onnx_path: Union[Path, str],
    output_onnx_path: Union[Path, str],
    keep_io_types: bool = True,
) -> Path:
    """Convert an FP32 ONNX model to half-precision FP16 weights.

    Args:
        input_onnx_path: Source FP32 .onnx model path.
        output_onnx_path: Target FP16 .onnx destination path.
        keep_io_types: Whether to preserve float32 input/output interfaces for compatibility.

    Returns:
        Path: Path to the generated FP16 ONNX model.
    """
    input_onnx_path = Path(input_onnx_path).resolve()
    output_onnx_path = Path(output_onnx_path).resolve()
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FP32 ONNX model from %s for FP16 conversion...", input_onnx_path)
    model = onnx.load(str(input_onnx_path), load_external_data=True)

    logger.info("Converting weights and intermediate operators to float16 (keep_io_types=%s)...", keep_io_types)
    model_fp16 = convert_float_to_float16(model, keep_io_types=keep_io_types)

    # Topologically sort graph to resolve cast node insertion ordering
    model_fp16 = _topological_sort_graph(model_fp16)

    onnx.save_model(model_fp16, str(output_onnx_path), save_as_external_data=False)
    # Clean up sidecar data if any
    sidecar_data = output_onnx_path.with_name(output_onnx_path.name + ".data")
    if sidecar_data.exists():
        sidecar_data.unlink()

    onnx.checker.check_model(onnx.load(str(output_onnx_path)))
    logger.info("Saved verified FP16 ONNX model to %s (size: %d bytes)", output_onnx_path, output_onnx_path.stat().st_size)
    return output_onnx_path


def quantize_static_int8(
    input_onnx_path: Union[Path, str],
    output_onnx_path: Union[Path, str],
    calibration_data: np.ndarray,
    quant_format: QuantFormat = QuantFormat.QDQ,
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QInt8,
) -> Path:
    """Perform post-training static INT8 quantization using internal validation calibration data.

    Args:
        input_onnx_path: Path to baseline FP32 ONNX model.
        output_onnx_path: Destination path for quantized INT8 model.
        calibration_data: Internal calibration samples (strictly internal, zero test data).
        quant_format: ONNX Runtime quantization format (default: QDQ).
        weight_type: Weight integer type (default: QInt8).
        activation_type: Activation integer type (default: QInt8).

    Returns:
        Path: Path to the generated static INT8 model.
    """
    input_onnx_path = Path(input_onnx_path).resolve()
    output_onnx_path = Path(output_onnx_path).resolve()
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing static INT8 quantization on %s with %d calibration samples...", input_onnx_path, len(calibration_data))
    reader = HARCalibrationDataReader(calibration_data)

    quantize_static(
        model_input=str(input_onnx_path),
        model_output=str(output_onnx_path),
        calibration_data_reader=reader,
        quant_format=quant_format,
        weight_type=weight_type,
        activation_type=activation_type,
    )

    # Ensure clean self-contained export
    model = onnx.load(str(output_onnx_path), load_external_data=True)
    onnx.save_model(model, str(output_onnx_path), save_as_external_data=False)
    sidecar_data = output_onnx_path.with_name(output_onnx_path.name + ".data")
    if sidecar_data.exists():
        sidecar_data.unlink()

    onnx.checker.check_model(onnx.load(str(output_onnx_path)))
    logger.info("Saved verified Static INT8 ONNX model to %s (size: %d bytes)", output_onnx_path, output_onnx_path.stat().st_size)
    return output_onnx_path


def quantize_dynamic_int8(
    input_onnx_path: Union[Path, str],
    output_onnx_path: Union[Path, str],
    weight_type: QuantType = QuantType.QInt8,
) -> Path:
    """Perform post-training dynamic INT8 quantization on model weights.

    Args:
        input_onnx_path: Path to baseline FP32 ONNX model.
        output_onnx_path: Destination path for dynamically quantized model.
        weight_type: Weight integer representation (default: QInt8).

    Returns:
        Path: Path to the generated dynamic INT8 model.
    """
    input_onnx_path = Path(input_onnx_path).resolve()
    output_onnx_path = Path(output_onnx_path).resolve()
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Performing dynamic INT8 quantization on %s...", input_onnx_path)
    quantize_dynamic(
        model_input=str(input_onnx_path),
        model_output=str(output_onnx_path),
        weight_type=weight_type,
    )

    model = onnx.load(str(output_onnx_path), load_external_data=True)
    onnx.save_model(model, str(output_onnx_path), save_as_external_data=False)
    sidecar_data = output_onnx_path.with_name(output_onnx_path.name + ".data")
    if sidecar_data.exists():
        sidecar_data.unlink()

    onnx.checker.check_model(onnx.load(str(output_onnx_path)))
    logger.info("Saved verified Dynamic INT8 ONNX model to %s (size: %d bytes)", output_onnx_path, output_onnx_path.stat().st_size)
    return output_onnx_path


def inspect_onnx_graph(onnx_path: Union[Path, str]) -> Dict[str, Any]:
    """Inspect the computational graph of an ONNX model and report operator distribution.

    Categorizes quantized versus floating-point operators and identifies operator-level quantization.

    Args:
        onnx_path: Path to the .onnx file to inspect.

    Returns:
        Dict[str, Any]: Comprehensive graph breakdown.
    """
    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

    model = onnx.load(str(onnx_path))
    graph = model.graph

    op_counts = Counter(node.op_type for node in graph.node)

    quantized_op_names = {
        "ConvInteger",
        "MatMulInteger",
        "QLinearConv",
        "QLinearMatMul",
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
    }

    quantized_nodes = sum(count for op, count in op_counts.items() if op in quantized_op_names)
    total_nodes = len(graph.node)
    floating_nodes = total_nodes - quantized_nodes

    conv_nodes_total = op_counts.get("Conv", 0) + op_counts.get("ConvInteger", 0) + op_counts.get("QLinearConv", 0)
    conv_quantized = op_counts.get("ConvInteger", 0) + op_counts.get("QLinearConv", 0)
    linear_quantized = op_counts.get("MatMulInteger", 0) + op_counts.get("QLinearMatMul", 0)

    # Classify precision
    if "ConvInteger" in op_counts or "MatMulInteger" in op_counts:
        precision_type = "int8_dynamic"
    elif "QuantizeLinear" in op_counts or "DequantizeLinear" in op_counts:
        precision_type = "int8_static_qdq"
    elif any("Float16" in str(init.data_type) for init in graph.initializer):
        precision_type = "fp16"
    else:
        precision_type = "fp32"

    file_size_bytes = onnx_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024**2), 4)

    return {
        "model_path": str(onnx_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "precision_type": precision_type,
        "total_nodes": total_nodes,
        "quantized_nodes": quantized_nodes,
        "floating_point_nodes": floating_nodes,
        "conv_layers_total": conv_nodes_total,
        "conv_layers_quantized": conv_quantized,
        "linear_layers_quantized": linear_quantized,
        "operator_distribution": dict(op_counts),
    }
