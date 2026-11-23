"""Comprehensive benchmark script for baseline PyTorch and optimized ONNX models.

Evaluates:
- Baseline PyTorch CNN (FP32)
- ONNX FP32
- ONNX FP16
- ONNX INT8 (Dynamic ConvInteger/MatMulInteger)
- ONNX INT8 (Static QDQ)

Across:
1. Internal Validation Partition (1,401 windows, 4 subjects [3, 11, 17, 27])
2. Official Test Partition (2,947 windows, 9 unseen subjects)
3. Inference Latency & Throughput (Batch-1 real-time edge profile)
4. Model Footprint & Parameter Breakdown
5. Confusion Matrices & Per-Class F1 Scores
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.preprocess import ChannelWiseStandardScaler
from src.deployment.benchmark import benchmark_latency, evaluate_classifier_predictions
from src.deployment.quantization import inspect_onnx_graph
from src.models.cnn_1d import ACTIVITY_NAMES, CNN1DHAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    split_name: str,
    output_path: Path,
) -> None:
    """Plot and save a normalized confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=ACTIVITY_NAMES,
        yticklabels=ACTIVITY_NAMES,
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"Normalized Confusion Matrix - {model_name} ({split_name})", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Activity Label", fontsize=10)
    ax.set_xlabel("Predicted Activity Label", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", output_path)


def get_pytorch_runner_and_predictor(model_pt_path: Path) -> Tuple[Callable, Callable]:
    """Create batch-1 runner and dataset predictor for PyTorch baseline."""
    har = CNN1DHAR.load(model_pt_path, device="cpu")
    net = har.model
    net.eval()

    def batch1_runner(x_sample: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t_in = torch.from_numpy(x_sample)
            return net(t_in).numpy()

    def dataset_predictor(X: np.ndarray) -> np.ndarray:
        # X is (N, 128, 9)
        return har.predict(X)

    return batch1_runner, dataset_predictor


def get_onnx_runner_and_predictor(onnx_path: Path) -> Tuple[Callable, Callable]:
    """Create batch-1 runner and dataset predictor for ONNX model."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def batch1_runner(x_sample: np.ndarray) -> np.ndarray:
        # x_sample is (1, 9, 128)
        return session.run([output_name], {input_name: x_sample})[0]

    def dataset_predictor(X: np.ndarray) -> np.ndarray:
        # X is (N, 128, 9)
        X_transposed = np.transpose(X, (0, 2, 1)).astype(np.float32)
        batch_size = 64
        n_samples = len(X_transposed)
        all_preds = []
        for i in range(0, n_samples, batch_size):
            chunk = X_transposed[i : i + batch_size]
            logits = session.run([output_name], {input_name: chunk})[0]
            preds = np.argmax(logits, axis=-1)
            all_preds.append(preds)
        return np.concatenate(all_preds, axis=0)

    return batch1_runner, dataset_predictor


def main() -> None:
    logger.info("=" * 80)
    logger.info("PHASE 4: BENCHMARK OPTIMIZED MODEL VARIANTS")
    logger.info("=" * 80)

    # 1. Load data
    logger.info("Loading unscaled sensor datasets and subject partitions...")
    raw_train_path = repo_root / "data" / "processed" / "X_train_raw.npy"
    y_train_path = repo_root / "data" / "processed" / "y_train.npy"
    subj_train_path = repo_root / "data" / "processed" / "subject_train.npy"

    raw_test_path = repo_root / "data" / "processed" / "X_test_raw.npy"
    y_test_path = repo_root / "data" / "processed" / "y_test.npy"
    subj_test_path = repo_root / "data" / "processed" / "subject_test.npy"

    X_train_raw = np.load(raw_train_path)
    y_train = np.load(y_train_path)
    subj_train = np.load(subj_train_path)

    X_test_raw = np.load(raw_test_path)
    y_test = np.load(y_test_path)
    subj_test = np.load(subj_test_path)

    # 2. Partition internal validation split (zero leakage)
    val_mask = np.isin(subj_train, [3, 11, 17, 27])
    internal_train_mask = ~val_mask

    val_scaler = ChannelWiseStandardScaler()
    val_scaler.fit(X_train_raw[internal_train_mask])
    X_val_scaled = val_scaler.transform(X_train_raw[val_mask])
    y_val = y_train[val_mask]

    # 3. Partition official test split (training scaler fitted on full 7,352 samples)
    official_scaler = ChannelWiseStandardScaler()
    official_scaler.fit(X_train_raw)
    X_test_scaled = official_scaler.transform(X_test_raw)

    logger.info("Internal Validation Set: %d windows (%d unseen subjects)", len(X_val_scaled), 4)
    logger.info("Official Test Set       : %d windows (%d unseen subjects)", len(X_test_scaled), 9)

    # Models directory
    models_dir = repo_root / "models"
    opt_dir = models_dir / "optimized"

    model_registry = [
        {
            "name": "PyTorch Baseline (FP32)",
            "type": "pytorch",
            "path": models_dir / "cnn_1d.pt",
            "precision": "FP32",
        },
        {
            "name": "ONNX (FP32)",
            "type": "onnx",
            "path": opt_dir / "cnn_1d_fp32.onnx",
            "precision": "FP32",
        },
        {
            "name": "ONNX (FP16)",
            "type": "onnx",
            "path": opt_dir / "cnn_1d_fp16.onnx",
            "precision": "FP16",
        },
        {
            "name": "ONNX INT8 (Dynamic)",
            "type": "onnx",
            "path": opt_dir / "cnn_1d_int8.onnx",
            "precision": "INT8 Dynamic",
        },
        {
            "name": "ONNX INT8 (Static QDQ)",
            "type": "onnx",
            "path": opt_dir / "cnn_1d_int8_static.onnx",
            "precision": "INT8 Static QDQ",
        },
    ]

    # Single dummy sample for batch-1 latency benchmark: shape (1, 9, 128)
    dummy_batch1 = np.random.randn(1, 9, 128).astype(np.float32)

    results_table: List[Dict[str, Any]] = []
    baseline_val_acc: float = 0.0
    baseline_test_acc: float = 0.0
    baseline_fp32_size: int = 0
    detailed_metrics: Dict[str, Any] = {}

    fig_dir = repo_root / "reports" / "day3_confusion_matrices"

    for entry in model_registry:
        m_name = entry["name"]
        m_path = entry["path"]
        m_type = entry["type"]
        m_precision = entry["precision"]

        logger.info("\n" + "=" * 60)
        logger.info("Benchmarking: %s", m_name)
        logger.info("Path        : %s", m_path)
        logger.info("=" * 60)

        if not m_path.exists():
            logger.warning("Model file %s does not exist! Skipping.", m_path)
            continue

        file_size_bytes = m_path.stat().st_size
        file_size_mb = round(file_size_bytes / (1024**2), 4)

        if m_name == "PyTorch Baseline (FP32)":
            baseline_fp32_size = file_size_bytes

        # Size reduction
        size_reduction_pct = 0.0
        if baseline_fp32_size > 0:
            size_reduction_pct = round(((baseline_fp32_size - file_size_bytes) / baseline_fp32_size) * 100.0, 2)

        # Create runners
        if m_type == "pytorch":
            b1_runner, predictor = get_pytorch_runner_and_predictor(m_path)
        else:
            b1_runner, predictor = get_onnx_runner_and_predictor(m_path)

        # 1. Latency benchmark (5 independent runs of 200 iterations = 1,000 timed samples)
        logger.info("Running Batch-1 real-time latency profiling (50 warm-up, 5 runs of 200 iterations = 1000 total)...")
        lat_stats = benchmark_latency(b1_runner, dummy_batch1, warm_up=50, repetitions=200, n_runs=5)

        # 2. Validation evaluation (1,401 samples)
        logger.info("Evaluating on internal validation set (1,401 samples)...")
        val_preds = predictor(X_val_scaled)
        val_metrics = evaluate_classifier_predictions(y_val, val_preds)

        if m_name == "PyTorch Baseline (FP32)":
            baseline_val_acc = val_metrics["accuracy_pct"]

        val_acc_delta = round(val_metrics["accuracy_pct"] - baseline_val_acc, 2)

        # 3. Test evaluation (2,947 samples)
        logger.info("Evaluating on official test set (2,947 samples)...")
        test_preds = predictor(X_test_scaled)
        test_metrics = evaluate_classifier_predictions(y_test, test_preds)

        if m_name == "PyTorch Baseline (FP32)":
            baseline_test_acc = test_metrics["accuracy_pct"]

        test_acc_delta = round(test_metrics["accuracy_pct"] - baseline_test_acc, 2)

        # Plot confusion matrices
        cm_val_path = fig_dir / f"cm_val_{m_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        cm_test_path = fig_dir / f"cm_test_{m_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plot_and_save_confusion_matrix(np.array(val_metrics["confusion_matrix"]), m_name, "Validation", cm_val_path)
        plot_and_save_confusion_matrix(np.array(test_metrics["confusion_matrix"]), m_name, "Official Test", cm_test_path)

        row = {
            "model": m_name,
            "precision": m_precision,
            "size_bytes": file_size_bytes,
            "size_mb": file_size_mb,
            "size_reduction_pct": size_reduction_pct,
            "val_accuracy_pct": val_metrics["accuracy_pct"],
            "val_macro_f1": val_metrics["f1_macro"],
            "val_acc_delta_pct": val_acc_delta,
            "test_accuracy_pct": test_metrics["accuracy_pct"],
            "test_macro_f1": test_metrics["f1_macro"],
            "test_weighted_f1": test_metrics["f1_weighted"],
            "test_acc_delta_pct": test_acc_delta,
            "latency_median_ms": lat_stats["median_latency_ms"],
            "latency_mean_ms": lat_stats["mean_latency_ms"],
            "latency_std_ms": lat_stats["std_latency_ms"],
            "inter_run_std_ms": lat_stats["inter_run_std_ms"],
            "latency_p95_ms": lat_stats["p95_latency_ms"],
            "latency_p99_ms": lat_stats["p99_latency_ms"],
            "throughput_windows_sec": lat_stats["throughput_windows_per_sec"],
        }
        results_table.append(row)

        detailed_metrics[m_name] = {
            "metadata": row,
            "latency_stats": lat_stats,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        logger.info(
            "[%s] Val Acc: %.2f%% (Δ: %+.2f%%) | Test Acc: %.2f%% (Δ: %+.2f%%) | Latency: %.3f ms | Size: %.2f MB (%+.1f%%)",
            m_name,
            val_metrics["accuracy_pct"],
            val_acc_delta,
            test_metrics["accuracy_pct"],
            test_acc_delta,
            lat_stats["median_latency_ms"],
            file_size_mb,
            size_reduction_pct,
        )

    # Convert to DataFrame and save CSV
    df_results = pd.DataFrame(results_table)
    csv_path = repo_root / "reports" / "day3_model_comparison.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info("Saved comparative benchmark table to %s", csv_path)

    # Save detailed JSON metrics
    json_path = repo_root / "reports" / "day3_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2)
    logger.info("Saved full metrics JSON to %s", json_path)

    # Save benchmark execution details
    benchmark_details = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measurement_function": "time.perf_counter()",
        "batch_size": 1,
        "warm_up_iterations": 50,
        "timed_iterations_per_run": 200,
        "benchmark_runs": 5,
        "total_timed_samples_per_model": 1000,
        "runtime_provider": "CPUExecutionProvider",
        "cpu_threads": torch.get_num_threads(),
        "throughput_definition": "Total items processed / total elapsed iteration time (batch_size / mean_latency_sec)",
        "statistical_caveat": (
            "P99 latency estimates over 1,000 samples represent extreme tail percentiles "
            "sensitive to OS scheduling jitter; evaluate alongside median latency and inter-run standard deviation."
        ),
        "models_evaluated": [m["name"] for m in model_registry],
        "validation_samples_count": len(X_val_scaled),
        "test_samples_count": len(X_test_scaled),
    }
    details_path = repo_root / "reports" / "day3_benchmark_details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_details, f, indent=2)
    logger.info("Saved benchmark details to %s", details_path)

    print("\n" + "=" * 90)
    print("DAY 3 OPTIMIZED MODEL COMPARISON TABLE")
    print("=" * 90)
    print(df_results.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
