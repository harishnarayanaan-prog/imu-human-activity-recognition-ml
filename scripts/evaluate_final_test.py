"""Evaluate frozen models on the official held-out test partition (2,947 windows)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, List, Any

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.evaluation.metrics import (
    compute_benchmark_metrics,
    plot_confusion_matrix,
    ACTIVITY_NAMES,
)
from src.models.random_forest import RandomForestHAR
from src.models.svm import SVMHAR
from src.models.cnn_1d import CNN1DHAR

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    repo_root = Path(__file__).resolve().parent.parent

    # Load frozen test set
    X_test_raw = np.load(repo_root / "data" / "processed" / "X_test.npy")
    X_test_feat = np.load(repo_root / "data" / "processed" / "X_test_features.npy")
    y_test = np.load(repo_root / "data" / "processed" / "y_test.npy")
    subj_test = np.load(repo_root / "data" / "processed" / "subject_test.npy")

    logger.info("Loaded held-out test set: %d windows across %d subjects", len(y_test), len(np.unique(subj_test)))
    logger.info("Test subjects: %s", sorted(list(np.unique(subj_test))))

    fig_dir = repo_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    # 1. Random Forest
    rf_path = repo_root / "models" / "random_forest.joblib"
    logger.info("Evaluating Random Forest on frozen test set...")
    rf = RandomForestHAR.load(rf_path)
    t0 = time.perf_counter()
    rf_preds = rf.predict(X_test_feat)
    rf_latency = (time.perf_counter() - t0) * 1000.0 / len(y_test)
    rf_metrics = compute_benchmark_metrics(
        y_test,
        rf_preds,
        latency_ms=rf_latency,
        training_time_sec=rf.training_time_sec_,
        model_size_mb=rf_path.stat().st_size / (1024 * 1024),
    )
    rf_metrics["model"] = "Random Forest"
    results.append(rf_metrics)
    plot_confusion_matrix(
        y_test,
        rf_preds,
        title="Random Forest: Final Test Confusion Matrix (2,947 windows)",
        save_path=fig_dir / "test_rf_confusion_matrix.png",
    )
    print("\n--- Random Forest Final Test Report ---")
    print(classification_report(y_test, rf_preds, target_names=ACTIVITY_NAMES, digits=4))

    # 2. Support Vector Machine
    svm_path = repo_root / "models" / "svm.joblib"
    logger.info("Evaluating SVM on frozen test set...")
    svm = SVMHAR.load(svm_path)
    t0 = time.perf_counter()
    svm_preds = svm.predict(X_test_feat)
    svm_latency = (time.perf_counter() - t0) * 1000.0 / len(y_test)
    svm_metrics = compute_benchmark_metrics(
        y_test,
        svm_preds,
        latency_ms=svm_latency,
        training_time_sec=svm.training_time_sec_,
        model_size_mb=svm_path.stat().st_size / (1024 * 1024),
    )
    svm_metrics["model"] = "SVM (RBF)"
    results.append(svm_metrics)
    plot_confusion_matrix(
        y_test,
        svm_preds,
        title="SVM: Final Test Confusion Matrix (2,947 windows)",
        save_path=fig_dir / "test_svm_confusion_matrix.png",
    )
    print("\n--- SVM (RBF) Final Test Report ---")
    print(classification_report(y_test, svm_preds, target_names=ACTIVITY_NAMES, digits=4))

    # 3. 1D CNN
    cnn_path = repo_root / "models" / "cnn_1d.pt"
    logger.info("Evaluating 1D CNN on frozen test set...")
    cnn = CNN1DHAR.load(cnn_path, device="cpu")
    t0 = time.perf_counter()
    cnn_preds = cnn.predict(X_test_raw)
    cnn_latency = (time.perf_counter() - t0) * 1000.0 / len(y_test)
    cnn_metrics = compute_benchmark_metrics(
        y_test,
        cnn_preds,
        latency_ms=cnn_latency,
        training_time_sec=cnn.training_time_sec_,
        model_size_mb=cnn_path.stat().st_size / (1024 * 1024),
        n_params=cnn.count_parameters(),
    )
    cnn_metrics["model"] = "1D CNN"
    results.append(cnn_metrics)
    plot_confusion_matrix(
        y_test,
        cnn_preds,
        title="1D CNN: Final Test Confusion Matrix (2,947 windows)",
        save_path=fig_dir / "test_cnn_confusion_matrix.png",
    )
    print("\n--- 1D CNN Final Test Report ---")
    print(classification_report(y_test, cnn_preds, target_names=ACTIVITY_NAMES, digits=4))

    # Assemble summary DataFrame
    df = pd.DataFrame(results)
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

    csv_path = repo_root / "reports" / "test_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved final test comparison to %s", csv_path)

    print("\n" + "=" * 80)
    print("FINAL TEST-SET BENCHMARK RESULTS (9 Held-Out Subjects, 2,947 Windows)")
    print("=" * 80)
    print(df[lead_cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
