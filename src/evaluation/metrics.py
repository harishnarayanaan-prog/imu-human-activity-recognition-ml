"""Comprehensive evaluation metrics, confusion matrices, and model comparison suite."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.data.splitting import subject_wise_train_val_split
from src.models.random_forest import RandomForestHAR
from src.models.svm import SVMHAR
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


def compute_benchmark_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latency_ms: float = 0.0,
    training_time_sec: float = 0.0,
    model_size_mb: float = 0.0,
    n_params: int = 0,
) -> Dict[str, Any]:
    """Compute standard classification and deployment metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_dict = {
        f"f1_{ACTIVITY_NAMES[i]}": round(float(per_class_f1[i]), 4)
        for i in range(len(ACTIVITY_NAMES))
    }

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "latency_ms": round(float(latency_ms), 4),
        "training_time_sec": round(float(training_time_sec), 2),
        "model_size_mb": round(float(model_size_mb), 2),
        "n_params": int(n_params),
        **per_class_dict,
    }
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ACTIVITY_NAMES,
    title: str = "Confusion Matrix",
    save_path: Optional[Path | str] = None,
) -> None:
    """Plot and save a clean normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Ground Truth Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info("Saved confusion matrix plot to %s", save_path)
    plt.close()


def plot_model_comparisons(df: pd.DataFrame, save_dir: Path | str) -> None:
    """Generate comparative visualization plots across models."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    models = df["model"].tolist()
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    # 1. Accuracy comparison
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, df["accuracy"] * 100, color=colors, edgecolor="black", width=0.55)
    plt.title("Model Validation Accuracy Comparison", fontsize=12, pad=12)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.ylim([90, 101])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "model_accuracy_comparison.png", dpi=300)
    plt.close()

    # 2. Macro F1 comparison
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, df["f1_macro"], color=colors, edgecolor="black", width=0.55)
    plt.title("Model Validation Macro F1 Comparison", fontsize=12, pad=12)
    plt.ylabel("Macro F1 Score", fontsize=10)
    plt.ylim([0.90, 1.01])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "macro_f1_comparison.png", dpi=300)
    plt.close()

    # 3. Training time comparison
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, df["training_time_sec"], color=colors, edgecolor="black", width=0.55)
    plt.title("Model Training Duration Comparison", fontsize=12, pad=12)
    plt.ylabel("Training Time (seconds)", fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.1f}s", ha="center", va="bottom", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "training_time_comparison.png", dpi=300)
    plt.close()

    # 4. Inference latency comparison
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, df["latency_ms"], color=colors, edgecolor="black", width=0.55)
    plt.title("Inference Latency per Window (2.56s)", fontsize=12, pad=12)
    plt.ylabel("Latency (ms / window)", fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f"{yval:.3f} ms", ha="center", va="bottom", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "inference_latency_comparison.png", dpi=300)
    plt.close()
    logger.info("Saved all model comparison figures to %s", save_dir)


def run_validation_comparison() -> pd.DataFrame:
    """Evaluate and compare all trained models on the subject-wise validation set."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Load data
    X_raw = np.load(repo_root / "data" / "processed" / "X_train.npy")
    X_feat = np.load(repo_root / "data" / "processed" / "X_train_features.npy")
    y = np.load(repo_root / "data" / "processed" / "y_train.npy")
    subjects = np.load(repo_root / "data" / "processed" / "subject_train.npy")

    # Splits
    split_raw = subject_wise_train_val_split(X_raw, y, subjects, val_subjects=[3, 11, 17, 27])
    split_feat = subject_wise_train_val_split(X_feat, y, subjects, val_subjects=[3, 11, 17, 27])

    y_val = split_raw.y_val
    fig_dir = repo_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    # 1. Random Forest
    rf_path = repo_root / "models" / "random_forest.joblib"
    logger.info("Evaluating Random Forest from %s...", rf_path.name)
    rf = RandomForestHAR.load(rf_path)
    t0 = time.perf_counter()
    rf_preds = rf.predict(split_feat.X_val)
    rf_latency = (time.perf_counter() - t0) * 1000.0 / len(y_val)
    rf_size_mb = rf_path.stat().st_size / (1024 * 1024)

    rf_metrics = compute_benchmark_metrics(
        y_val,
        rf_preds,
        latency_ms=rf_latency,
        training_time_sec=rf.training_time_sec_,
        model_size_mb=rf_size_mb,
        n_params=0,
    )
    rf_metrics["model"] = "Random Forest"
    results.append(rf_metrics)
    plot_confusion_matrix(
        y_val,
        rf_preds,
        title="Random Forest: Validation Confusion Matrix",
        save_path=fig_dir / "rf_confusion_matrix.png",
    )

    # 2. Support Vector Machine (SVM)
    svm_path = repo_root / "models" / "svm.joblib"
    logger.info("Evaluating SVM from %s...", svm_path.name)
    svm = SVMHAR.load(svm_path)
    t0 = time.perf_counter()
    svm_preds = svm.predict(split_feat.X_val)
    svm_latency = (time.perf_counter() - t0) * 1000.0 / len(y_val)
    svm_size_mb = svm_path.stat().st_size / (1024 * 1024)

    svm_metrics = compute_benchmark_metrics(
        y_val,
        svm_preds,
        latency_ms=svm_latency,
        training_time_sec=svm.training_time_sec_,
        model_size_mb=svm_size_mb,
        n_params=0,
    )
    svm_metrics["model"] = "SVM (RBF)"
    results.append(svm_metrics)
    plot_confusion_matrix(
        y_val,
        svm_preds,
        title="SVM: Validation Confusion Matrix",
        save_path=fig_dir / "svm_confusion_matrix.png",
    )

    # 3. 1D CNN
    cnn_path = repo_root / "models" / "cnn_1d.pt"
    logger.info("Evaluating 1D CNN from %s...", cnn_path.name)
    cnn = CNN1DHAR.load(cnn_path, device="cpu")
    t0 = time.perf_counter()
    cnn_preds = cnn.predict(split_raw.X_val)
    cnn_latency = (time.perf_counter() - t0) * 1000.0 / len(y_val)
    cnn_size_mb = cnn_path.stat().st_size / (1024 * 1024)

    cnn_metrics = compute_benchmark_metrics(
        y_val,
        cnn_preds,
        latency_ms=cnn_latency,
        training_time_sec=cnn.training_time_sec_,
        model_size_mb=cnn_size_mb,
        n_params=cnn.count_parameters(),
    )
    cnn_metrics["model"] = "1D CNN"
    results.append(cnn_metrics)
    plot_confusion_matrix(
        y_val,
        cnn_preds,
        title="1D CNN: Validation Confusion Matrix",
        save_path=fig_dir / "cnn_confusion_matrix.png",
    )

    # Assemble summary DataFrame
    df = pd.DataFrame(results)
    # Reorder columns
    lead_cols = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "latency_ms",
        "training_time_sec",
        "model_size_mb",
        "n_params",
    ]
    other_cols = [c for c in df.columns if c not in lead_cols]
    df = df[lead_cols + other_cols]

    csv_path = repo_root / "reports" / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Exported validation comparison table to %s", csv_path)

    # Generate comparison figures
    plot_model_comparisons(df, save_dir=fig_dir)

    print("\n" + "=" * 80)
    print("MODEL VALIDATION PERFORMANCE COMPARISON (Subject-Wise Validation Split)")
    print("=" * 80)
    print(df[lead_cols].to_string(index=False))
    print("=" * 80)

    return df


def main() -> None:
    """Run validation benchmark suite."""
    run_validation_comparison()


if __name__ == "__main__":
    main()
