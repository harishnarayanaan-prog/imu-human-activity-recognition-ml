"""PyTorch to ONNX model export and numerical parity verification.

Exports the Day 2 trained 1D CNN architecture (Conv1DNet) to ONNX format,
validates structural graph integrity with onnx.checker, and verifies numerical parity
(logits, probabilities, and class predictions) against PyTorch CPU inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.cnn_1d import CNN1DHAR, Conv1DNet

logger = logging.getLogger(__name__)


def export_pytorch_to_onnx(
    model_path: Union[Path, str],
    output_path: Union[Path, str],
    opset_version: int = 17,
    dynamic_batch: bool = True,
    try_dynamo: bool = True,
    embed_weights: bool = True,
) -> Path:
    """Export trained PyTorch CNN checkpoint to an optimized ONNX model file.

    Args:
        model_path: Path to the trained PyTorch checkpoint (.pt).
        output_path: Destination path for the exported .onnx file.
        opset_version: Target ONNX opset version (default: 17).
        dynamic_batch: Whether to assign dynamic batch axes to input and output.
        try_dynamo: Whether to attempt the modern TorchDynamo exporter first.
        embed_weights: Whether to embed all tensor weights directly into a single
            self-contained .onnx file (avoiding external .data files).

    Returns:
        Path: Resolved absolute path to the exported ONNX model.
    """
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found at: {model_path}")

    logger.info("Loading PyTorch model from %s", model_path)
    har_model = CNN1DHAR.load(model_path, device="cpu")
    net = har_model.model
    net.eval()

    # Input tensor shape: (batch_size, in_channels=9, seq_len=128)
    dummy_input = torch.randn(1, 9, 128, dtype=torch.float32)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "sensor_window": {0: "batch_size"},
            "activity_logits": {0: "batch_size"},
        }

    exported = False
    if try_dynamo:
        try:
            logger.info("Attempting PyTorch export with dynamo=True...")
            torch.onnx.export(
                net,
                dummy_input,
                str(output_path),
                export_params=True,
                input_names=["sensor_window"],
                output_names=["activity_logits"],
                dynamic_axes=dynamic_axes,
                dynamo=True,
            )
            exported = True
            logger.info("ONNX export with dynamo=True completed successfully.")
        except Exception as exc:
            logger.warning(
                "TorchDynamo exporter encountered an issue (%s: %s). "
                "Falling back to standard TorchScript-based exporter.",
                type(exc).__name__,
                exc,
            )

    if not exported:
        logger.info(
            "Exporting using standard PyTorch ONNX exporter (opset=%d, dynamic_batch=%s)...",
            opset_version,
            dynamic_batch,
        )
        torch.onnx.export(
            net,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["sensor_window"],
            output_names=["activity_logits"],
            dynamic_axes=dynamic_axes,
        )
        logger.info("Standard ONNX export completed successfully.")

    # Clean up and embed external data if generated, ensuring a single portable file
    if embed_weights:
        onnx_model = onnx.load(str(output_path), load_external_data=True)
        onnx.save_model(onnx_model, str(output_path), save_as_external_data=False)
        # Remove any external data sidecar file
        sidecar_data = output_path.with_name(output_path.name + ".data")
        if sidecar_data.exists():
            sidecar_data.unlink()

    # Structural validation
    logger.info("Validating exported ONNX graph structure with onnx.checker...")
    final_model = onnx.load(str(output_path))
    onnx.checker.check_model(final_model)
    logger.info("ONNX structural checker passed without errors.")

    return output_path


def verify_numerical_parity(
    model_pt_path: Union[Path, str],
    model_onnx_path: Union[Path, str],
    test_samples: np.ndarray,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """Verify numerical parity between PyTorch CPU inference and ONNX Runtime.

    Compares raw unscaled logits, softmax probabilities, and argmax activity predictions.

    Args:
        model_pt_path: Path to PyTorch .pt checkpoint.
        model_onnx_path: Path to exported .onnx model file.
        test_samples: Array of input windows of shape (N, 128, 9) or (N, 9, 128).
        atol: Maximum acceptable absolute difference for logits (default: 1e-4).

    Returns:
        Dict[str, Any]: Detailed parity comparison results.
    """
    model_pt_path = Path(model_pt_path).resolve()
    model_onnx_path = Path(model_onnx_path).resolve()

    if not model_pt_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {model_pt_path}")
    if not model_onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_onnx_path}")

    # Prepare input tensor: if (N, 128, 9), transpose to (N, 9, 128)
    if test_samples.ndim == 2:
        test_samples = np.expand_dims(test_samples, axis=0)
    if test_samples.shape[1] == 128 and test_samples.shape[2] == 9:
        X_transposed = np.transpose(test_samples, (0, 2, 1)).astype(np.float32)
    elif test_samples.shape[1] == 9 and test_samples.shape[2] == 128:
        X_transposed = test_samples.astype(np.float32)
    else:
        raise ValueError(
            f"Invalid test sample shape: {test_samples.shape}. Expected (N, 128, 9) or (N, 9, 128)."
        )

    # 1. PyTorch inference
    har_model = CNN1DHAR.load(model_pt_path, device="cpu")
    net = har_model.model
    net.eval()
    with torch.no_grad():
        pt_tensor = torch.from_numpy(X_transposed)
        pt_logits = net(pt_tensor).cpu().numpy()
        pt_probs = torch.softmax(torch.from_numpy(pt_logits), dim=-1).numpy()
        pt_preds = np.argmax(pt_logits, axis=-1)

    # 2. ONNX Runtime inference
    session = ort.InferenceSession(str(model_onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    onnx_logits = session.run([output_name], {input_name: X_transposed})[0]
    # Compute softmax over logits
    exp_logits = np.exp(onnx_logits - np.max(onnx_logits, axis=-1, keepdims=True))
    onnx_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    onnx_preds = np.argmax(onnx_logits, axis=-1)

    # 3. Compute metrics
    max_logit_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
    mean_logit_diff = float(np.mean(np.abs(pt_logits - onnx_logits)))
    max_prob_diff = float(np.max(np.abs(pt_probs - onnx_probs)))
    mean_prob_diff = float(np.mean(np.abs(pt_probs - onnx_probs)))
    matching_predictions = int(np.sum(pt_preds == onnx_preds))
    total_samples = int(len(pt_preds))
    agreement_pct = float((matching_predictions / total_samples) * 100.0)

    parity_passed = bool((max_logit_diff <= atol) and (agreement_pct == 100.0))

    results = {
        "num_samples": total_samples,
        "max_absolute_logit_diff": max_logit_diff,
        "mean_absolute_logit_diff": mean_logit_diff,
        "max_prob_diff": max_prob_diff,
        "mean_prob_diff": mean_prob_diff,
        "prediction_agreement_pct": agreement_pct,
        "matching_predictions": matching_predictions,
        "tolerance_threshold": atol,
        "parity_passed": parity_passed,
    }

    logger.info(
        "Parity Check: Samples=%d, Max Logit Diff=%.6e, Mean Logit Diff=%.6e, "
        "Max Prob Diff=%.6e, Agreement=%.2f%%, Status=%s",
        total_samples,
        max_logit_diff,
        mean_logit_diff,
        max_prob_diff,
        agreement_pct,
        "PASSED" if parity_passed else "FAILED",
    )

    return results
