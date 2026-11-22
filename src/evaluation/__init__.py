"""Evaluation metrics, profiling, and comparison utilities for HAR models."""

from src.evaluation.metrics import (
    compute_benchmark_metrics,
    plot_confusion_matrix,
    plot_model_comparisons,
    run_validation_comparison,
)

__all__ = [
    "compute_benchmark_metrics",
    "plot_confusion_matrix",
    "plot_model_comparisons",
    "run_validation_comparison",
]
