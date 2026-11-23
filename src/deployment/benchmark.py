"""Benchmarking module for inference latency, throughput, and classification performance.

Evaluates baseline PyTorch and optimized ONNX variants on internal validation data
and held-out official test data under standardized CPU execution environments.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch

from src.models.cnn_1d import CNN1DHAR

logger = logging.getLogger(__name__)

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def benchmark_latency(
    inference_fn: Callable[[np.ndarray], Any],
    input_sample: np.ndarray,
    warm_up: int = 50,
    repetitions: int = 200,
) -> Dict[str, float]:
    """Measure inference latency and throughput for an arbitrary callable runner.

    Args:
        inference_fn: Callable accepting input_sample and returning model predictions/logits.
        input_sample: Array of shape (B, 9, 128) or (B, 128, 9).
        warm_up: Number of untimed warm-up iterations (default: 50).
        repetitions: Number of timed benchmark iterations (default: 200).

    Returns:
        Dict[str, float]: Detailed latency statistics in milliseconds and throughput in windows/sec.
    """
    batch_size = input_sample.shape[0] if input_sample.ndim >= 2 else 1

    # 1. Warm-up phase
    for _ in range(warm_up):
        _ = inference_fn(input_sample)

    # 2. Timed benchmark phase
    latencies_sec: List[float] = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        _ = inference_fn(input_sample)
        t1 = time.perf_counter()
        latencies_sec.append(t1 - t0)

    latencies_ms = np.array(latencies_sec) * 1000.0

    mean_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p90_ms = float(np.percentile(latencies_ms, 90))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    std_ms = float(np.std(latencies_ms))

    # Throughput in processed windows per second
    throughput = float((batch_size * repetitions) / np.sum(latencies_sec))

    return {
        "warm_up_iterations": warm_up,
        "timed_iterations": repetitions,
        "batch_size": batch_size,
        "mean_latency_ms": round(mean_ms, 4),
        "median_latency_ms": round(median_ms, 4),
        "p90_latency_ms": round(p90_ms, 4),
        "p95_latency_ms": round(p95_ms, 4),
        "p99_latency_ms": round(p99_ms, 4),
        "min_latency_ms": round(min_ms, 4),
        "max_latency_ms": round(max_ms, 4),
        "std_latency_ms": round(std_ms, 4),
        "throughput_windows_per_sec": round(throughput, 2),
    }


def evaluate_classifier_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics against ground-truth labels.

    Args:
        y_true: True activity labels (0-indexed).
        y_pred: Predicted activity labels (0-indexed).

    Returns:
        Dict[str, Any]: Accuracy, precision, recall, macro/weighted F1, and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ACTIVITY_NAMES))))

    # Per-class F1-scores
    per_class_f1: Dict[str, float] = {}
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    for idx, act in enumerate(ACTIVITY_NAMES):
        per_class_f1[act] = round(float(f1_per[idx]), 4) if idx < len(f1_per) else 0.0

    return {
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100.0, 2),
        "precision_macro": round(p_macro, 4),
        "recall_macro": round(r_macro, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_f1": per_class_f1,
    }


evaluate_classifier = evaluate_classifier_predictions


def create_onnx_predictor(
    onnx_path: Union[Path, str],
    batch_size: int = 64,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a batched prediction function for an ONNX model file.

    Args:
        onnx_path: Path to the .onnx model.
        batch_size: Chunk size for evaluating large arrays.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Predictor function taking (N, 128, 9) or (N, 9, 128)
            and returning (N,) integer class predictions (0..5).
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def predictor(X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        # Transpose to (N, 9, 128) if (N, 128, 9)
        if X.shape[1] == 128 and X.shape[2] == 9:
            X_in = np.transpose(X, (0, 2, 1)).astype(np.float32)
        elif X.shape[1] == 9 and X.shape[2] == 128:
            X_in = X.astype(np.float32)
        else:
            raise ValueError(f"Unexpected shape: {X.shape}. Expected (N, 128, 9) or (N, 9, 128).")

        n_samples = len(X_in)
        all_preds: List[np.ndarray] = []

        for start in range(0, n_samples, batch_size):
            chunk = X_in[start : start + batch_size]
            logits = session.run([output_name], {input_name: chunk})[0]
            preds = np.argmax(logits, axis=-1)
            all_preds.append(preds)

        return np.concatenate(all_preds, axis=0)

    return predictor
