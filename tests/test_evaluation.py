"""Unit tests for evaluation metrics and reporting."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_benchmark_metrics,
    plot_confusion_matrix,
    ACTIVITY_NAMES,
)


def test_compute_benchmark_metrics() -> None:
    """Verify compute_benchmark_metrics dictionary output structure."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 0, 1, 3, 3])

    metrics = compute_benchmark_metrics(
        y_true,
        y_pred,
        latency_ms=0.05,
        training_time_sec=1.2,
        model_size_mb=3.4,
        n_params=1000,
    )

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "latency_ms" in metrics
    assert "training_time_sec" in metrics
    assert "model_size_mb" in metrics
    assert "n_params" in metrics
    assert metrics["accuracy"] == 0.9  # 9 / 10 correct
    assert metrics["latency_ms"] == 0.05
    assert metrics["n_params"] == 1000

    for name in ACTIVITY_NAMES:
        assert f"f1_{name}" in metrics


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    """Verify plot_confusion_matrix writes image to disk."""
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([0, 1, 2, 3, 4, 5])

    save_path = tmp_path / "test_cm.png"
    plot_confusion_matrix(y_true, y_pred, title="Test Matrix", save_path=save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 1000
