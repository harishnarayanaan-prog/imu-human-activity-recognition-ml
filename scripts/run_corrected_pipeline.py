"""Master script to execute Day 2.1 Methodology-Corrected Pipeline.

Executes:
1. Phase 2: Leak-free validation pipeline (zero subject overlap, scaler fitted strictly on internal train).
2. Phase 3: Final model training on full official training set and re-evaluation on official frozen test set.
3. Generates all metrics, tables, reports, and confusion matrices.
"""

from __future__ import annotations

import copy
import json
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.data.preprocess import ChannelWiseStandardScaler
from src.data.splitting import (
    subject_wise_train_val_split,
    prepare_leak_free_train_val_tensors,
    prepare_leak_free_train_val_features,
)
from src.features.feature_engineering import IMUFeatureExtractor
from src.models.random_forest import RandomForestHAR, plot_feature_importance
from src.models.svm import SVMHAR
from src.models.cnn_1d import CNN1DHAR
from src.evaluation.metrics import (
    compute_benchmark_metrics,
    plot_confusion_matrix,
    plot_model_comparisons,
    ACTIVITY_NAMES,
)

logger = logging.getLogger(__name__)


def run_corrected_validation(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    fig_dir: Path,
) -> pd.DataFrame:
    """Execute Phase 2: Leak-free validation pipeline."""
    logger.info("=" * 80)
    logger.info("PHASE 2: EXECUTING LEAK-FREE VALIDATION PIPELINE")
    logger.info("=" * 80)

    val_subjects = [3, 11, 17, 27]

    # 1. Leak-free tensors (scaler fitted strictly on 17 internal train subjects)
    (
        X_train_scaled,
        y_train_internal,
        X_val_scaled,
        y_val,
        val_scaler,
        split_tensor,
    ) = prepare_leak_free_train_val_tensors(
        X_train_raw, y_train, subj_train, val_subjects=val_subjects
    )

    # 2. Leak-free features (extracted from leak-free scaled tensors)
    extractor = IMUFeatureExtractor(sampling_rate_hz=50.0)
    logger.info("Extracting features from internal train windows (5,951 samples)...")
    X_train_feat = extractor.transform(X_train_scaled)
    logger.info("Extracting features from validation windows (1,401 samples)...")
    X_val_feat = extractor.transform(X_val_scaled)

    results: List[Dict[str, Any]] = []

    # Model A: Random Forest
    logger.info("Training Random Forest on leak-free internal train features...")
    rf_val = RandomForestHAR(n_estimators=150, max_depth=25, random_state=42, n_jobs=1)
    rf_val.fit(X_train_feat, y_train_internal)
    t0 = time.perf_counter()
    rf_val_preds = rf_val.predict(X_val_feat)
    rf_val_lat = (time.perf_counter() - t0) * 1000.0 / len(y_val)

    # Measure actual serialized size
    val_rf_path = repo_root / "models" / "rf_val.joblib"
    val_rf_path.parent.mkdir(parents=True, exist_ok=True)
    rf_val.save(val_rf_path)
    rf_val_size_mb = val_rf_path.stat().st_size / (1024 * 1024)
    val_rf_path.unlink(missing_ok=True)

    rf_metrics = compute_benchmark_metrics(
        y_val,
        rf_val_preds,
        latency_ms=rf_val_lat,
        training_time_sec=rf_val.training_time_sec_,
        model_size_mb=rf_val_size_mb,
        n_params=0,
    )
    rf_metrics["model"] = "Random Forest"
    results.append(rf_metrics)
    plot_confusion_matrix(
        y_val,
        rf_val_preds,
        title="Random Forest: Corrected Validation Confusion Matrix",
        save_path=fig_dir / "rf_confusion_matrix.png",
    )

    # Model B: Support Vector Machine (RBF)
    logger.info("Training SVM on leak-free internal train features...")
    svm_val = SVMHAR(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
    svm_val.fit(X_train_feat, y_train_internal)
    t0 = time.perf_counter()
    svm_val_preds = svm_val.predict(X_val_feat)
    svm_val_lat = (time.perf_counter() - t0) * 1000.0 / len(y_val)

    val_svm_path = repo_root / "models" / "svm_val.joblib"
    svm_val.save(val_svm_path)
    svm_val_size_mb = val_svm_path.stat().st_size / (1024 * 1024)
    val_svm_path.unlink(missing_ok=True)

    svm_metrics = compute_benchmark_metrics(
        y_val,
        svm_val_preds,
        latency_ms=svm_val_lat,
        training_time_sec=svm_val.training_time_sec_,
        model_size_mb=svm_val_size_mb,
        n_params=0,
    )
    svm_metrics["model"] = "SVM (RBF)"
    results.append(svm_metrics)
    plot_confusion_matrix(
        y_val,
        svm_val_preds,
        title="SVM: Corrected Validation Confusion Matrix",
        save_path=fig_dir / "svm_confusion_matrix.png",
    )

    # Model C: 1D Deep CNN (PyTorch)
    logger.info("Training 1D Deep CNN on leak-free internal train windows...")
    cnn_val = CNN1DHAR(
        in_channels=9,
        num_classes=6,
        batch_size=64,
        lr=0.001,
        max_epochs=35,
        patience=8,
        random_seed=42,
    )
    cnn_val.fit(X_train_scaled, y_train_internal, X_val_scaled, y_val)
    t0 = time.perf_counter()
    cnn_val_preds = cnn_val.predict(X_val_scaled)
    cnn_val_lat = (time.perf_counter() - t0) * 1000.0 / len(y_val)

    val_cnn_path = repo_root / "models" / "cnn_val.pt"
    cnn_val.save(val_cnn_path)
    cnn_val_size_mb = val_cnn_path.stat().st_size / (1024 * 1024)
    val_cnn_path.unlink(missing_ok=True)

    cnn_metrics = compute_benchmark_metrics(
        y_val,
        cnn_val_preds,
        latency_ms=cnn_val_lat,
        training_time_sec=cnn_val.training_time_sec_,
        model_size_mb=cnn_val_size_mb,
        n_params=cnn_val.count_parameters(),
    )
    cnn_metrics["model"] = "1D CNN"
    results.append(cnn_metrics)
    plot_confusion_matrix(
        y_val,
        cnn_val_preds,
        title="1D CNN: Corrected Validation Confusion Matrix",
        save_path=fig_dir / "cnn_confusion_matrix.png",
    )

    df_val = pd.DataFrame(results)
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
    other_cols = [c for c in df_val.columns if c not in lead_cols]
    df_val = df_val[lead_cols + other_cols]

    # Save comparison files
    df_val.to_csv(repo_root / "reports" / "model_comparison.csv", index=False)
    df_val.to_csv(repo_root / "reports" / "corrected_validation_comparison.csv", index=False)
    plot_model_comparisons(df_val, save_dir=fig_dir)

    # Also save feature importance plot
    importances = rf_val.get_feature_importances()
    plot_feature_importance(importances, top_k=20, save_path=fig_dir / "rf_feature_importance.png")

    logger.info("\n" + "=" * 80)
    logger.info("CORRECTED VALIDATION PERFORMANCE RESULTS (Zero Leakage)")
    logger.info("=" * 80)
    print(df_val[lead_cols].to_string(index=False))
    logger.info("=" * 80)

    return df_val, cnn_val.best_epoch_


def run_corrected_final_training(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    subj_test: np.ndarray,
    best_cnn_epochs: int,
    fig_dir: Path,
) -> pd.DataFrame:
    """Execute Phase 3: Final Model Training on full official train & test evaluation."""
    logger.info("=" * 80)
    logger.info("PHASE 3: FINAL MODEL TRAINING (All 21 Official Training Subjects)")
    logger.info("=" * 80)

    # Validate best_cnn_epochs upfront: prohibit silent fallback to default epochs
    if best_cnn_epochs is None or not isinstance(best_cnn_epochs, int) or best_cnn_epochs <= 0:
        raise ValueError(
            f"Invalid validation-derived epoch count: best_cnn_epochs={best_cnn_epochs!r}. "
            "Final CNN training requires an explicit, positive integer epoch budget derived from a "
            "successful internal validation run. Silent fallback is strictly prohibited to prevent "
            "concealing upstream validation failures."
        )

    # 1. Fit final scaler on ALL official training subjects
    final_scaler = ChannelWiseStandardScaler()
    final_scaler.fit(X_train_raw)
    X_train_scaled = final_scaler.transform(X_train_raw).astype(np.float32)
    X_test_scaled = final_scaler.transform(X_test_raw).astype(np.float32)

    # Save official scaled tensors
    proc_dir = repo_root / "data" / "processed"
    np.save(proc_dir / "X_train.npy", X_train_scaled)
    np.save(proc_dir / "X_test.npy", X_test_scaled)
    logger.info("Saved official scaled tensors to X_train.npy and X_test.npy")

    # 2. Extract features
    extractor = IMUFeatureExtractor(sampling_rate_hz=50.0)
    logger.info("Extracting 204 features from full official train partition (7,352 samples)...")
    X_train_feat = extractor.transform(X_train_scaled)
    logger.info("Extracting 204 features from official test partition (2,947 samples)...")
    X_test_feat = extractor.transform(X_test_scaled)

    np.save(proc_dir / "X_train_features.npy", X_train_feat)
    np.save(proc_dir / "X_test_features.npy", X_test_feat)
    logger.info("Saved official feature matrices to X_train_features.npy and X_test_features.npy")

    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    # Final Model A: Random Forest (7,352 samples)
    logger.info("Training Final Random Forest on full official training set...")
    rf_final = RandomForestHAR(n_estimators=150, max_depth=25, random_state=42, n_jobs=1)
    rf_final.fit(X_train_feat, y_train)
    rf_final.save(models_dir / "random_forest.joblib")

    t0 = time.perf_counter()
    rf_test_preds = rf_final.predict(X_test_feat)
    rf_test_lat = (time.perf_counter() - t0) * 1000.0 / len(y_test)
    rf_metrics = compute_benchmark_metrics(
        y_test,
        rf_test_preds,
        latency_ms=rf_test_lat,
        training_time_sec=rf_final.training_time_sec_,
        model_size_mb=(models_dir / "random_forest.joblib").stat().st_size / (1024 * 1024),
        n_params=0,
    )
    rf_metrics["model"] = "Random Forest"
    results.append(rf_metrics)
    plot_confusion_matrix(
        y_test,
        rf_test_preds,
        title="Random Forest: Final Test Confusion Matrix (Methodology Corrected)",
        save_path=fig_dir / "test_rf_confusion_matrix.png",
    )
    print("\n--- Final Test Report: Random Forest ---")
    print(classification_report(y_test, rf_test_preds, target_names=ACTIVITY_NAMES, digits=4))

    # Final Model B: SVM (7,352 samples)
    logger.info("Training Final SVM on full official training set...")
    svm_final = SVMHAR(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
    svm_final.fit(X_train_feat, y_train)
    svm_final.save(models_dir / "svm.joblib")

    t0 = time.perf_counter()
    svm_test_preds = svm_final.predict(X_test_feat)
    svm_test_lat = (time.perf_counter() - t0) * 1000.0 / len(y_test)
    svm_metrics = compute_benchmark_metrics(
        y_test,
        svm_test_preds,
        latency_ms=svm_test_lat,
        training_time_sec=svm_final.training_time_sec_,
        model_size_mb=(models_dir / "svm.joblib").stat().st_size / (1024 * 1024),
        n_params=0,
    )
    svm_metrics["model"] = "SVM (RBF)"
    results.append(svm_metrics)
    plot_confusion_matrix(
        y_test,
        svm_test_preds,
        title="SVM: Final Test Confusion Matrix (Methodology Corrected)",
        save_path=fig_dir / "test_svm_confusion_matrix.png",
    )
    print("\n--- Final Test Report: SVM (RBF) ---")
    print(classification_report(y_test, svm_test_preds, target_names=ACTIVITY_NAMES, digits=4))

    # Final Model C: 1D Deep CNN (7,352 samples)
    # Fixed epoch count validated upfront from the internal validation run.
    fixed_epochs = best_cnn_epochs

    # Save authoritative epoch selection metadata
    epoch_metadata = {
        "validation_selected_epoch": fixed_epochs,
        "selection_source": "internal_subject_wise_validation",
        "validation_subjects": [3, 11, 17, 27],
        "training_subjects_count": 17,
        "evaluation_protocol": "fixed_epochs_zero_test_leakage",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = repo_root / "reports" / "epoch_selection_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(epoch_metadata, f, indent=2)

    logger.info(
        "Training Final 1D CNN on full official training set (fixed epochs: %d, verified from validation metadata)...",
        fixed_epochs,
    )
    cnn_final = CNN1DHAR(
        in_channels=9,
        num_classes=6,
        batch_size=64,
        lr=0.001,
        random_seed=42,
    )
    # Train strictly on full official training windows with fixed epochs; NO test data passed
    cnn_final.fit_fixed_epochs(X_train_scaled, y_train, epochs=fixed_epochs)
    cnn_final.save(models_dir / "cnn_1d.pt")

    t0 = time.perf_counter()
    cnn_test_preds = cnn_final.predict(X_test_scaled)
    cnn_test_lat = (time.perf_counter() - t0) * 1000.0 / len(y_test)
    cnn_metrics = compute_benchmark_metrics(
        y_test,
        cnn_test_preds,
        latency_ms=cnn_test_lat,
        training_time_sec=cnn_final.training_time_sec_,
        model_size_mb=(models_dir / "cnn_1d.pt").stat().st_size / (1024 * 1024),
        n_params=cnn_final.count_parameters(),
    )
    cnn_metrics["model"] = "1D CNN"
    results.append(cnn_metrics)
    plot_confusion_matrix(
        y_test,
        cnn_test_preds,
        title="1D CNN: Final Test Confusion Matrix (Methodology Corrected)",
        save_path=fig_dir / "test_cnn_confusion_matrix.png",
    )
    print("\n--- Final Test Report: 1D CNN ---")
    print(classification_report(y_test, cnn_test_preds, target_names=ACTIVITY_NAMES, digits=4))

    # Assemble test summary DataFrame
    df_test = pd.DataFrame(results)
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
    other_cols = [c for c in df_test.columns if c not in lead_cols]
    df_test = df_test[lead_cols + other_cols]

    # Save test comparison files
    df_test.to_csv(repo_root / "reports" / "test_model_comparison.csv", index=False)
    df_test.to_csv(repo_root / "reports" / "corrected_test_comparison.csv", index=False)

    logger.info("\n" + "=" * 80)
    logger.info("CORRECTED FINAL TEST-SET BENCHMARK RESULTS (9 Held-Out Subjects, 2,947 Windows)")
    logger.info("=" * 80)
    print(df_test[lead_cols].to_string(index=False))
    logger.info("=" * 80)

    return df_test


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load unscaled raw data and labels
    proc_dir = repo_root / "data" / "processed"
    X_train_raw = np.load(proc_dir / "X_train_raw.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    subj_train = np.load(proc_dir / "subject_train.npy")

    X_test_raw = np.load(proc_dir / "X_test_raw.npy")
    y_test = np.load(proc_dir / "y_test.npy")
    subj_test = np.load(proc_dir / "subject_test.npy")

    fig_dir = repo_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Run Phase 2
    df_val, best_cnn_epoch = run_corrected_validation(
        X_train_raw, y_train, subj_train, fig_dir
    )

    # Run Phase 3
    df_test = run_corrected_final_training(
        X_train_raw,
        y_train,
        subj_train,
        X_test_raw,
        y_test,
        subj_test,
        best_cnn_epoch,
        fig_dir,
    )


if __name__ == "__main__":
    main()
