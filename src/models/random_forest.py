"""Random Forest Classifier for IMU Human Activity Recognition.

Trained on 204 engineered statistical and spectral features with subject-wise
cross-validation and evaluation.
"""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.data.splitting import subject_wise_train_val_split
from src.features.feature_engineering import get_feature_names

logger = logging.getLogger(__name__)

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


class RandomForestHAR:
    """Random Forest model wrapper for HAR classification."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: Optional[int] = 25,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.training_time_sec_: float = 0.0
        self.feature_names_: List[str] = get_feature_names()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestHAR":
        """Fit Random Forest on feature matrix."""
        start_t = time.perf_counter()
        self.model.fit(X, y)
        self.training_time_sec_ = time.perf_counter() - start_t
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict activity labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> List[Tuple[str, float]]:
        """Return sorted list of (feature_name, importance) tuples."""
        importances = self.model.feature_importances_
        paired = list(zip(self.feature_names_, importances))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired

    def save(self, filepath: Path | str) -> None:
        """Serialize model to disk using joblib."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved RandomForestHAR to %s", filepath)

    @classmethod
    def load(cls, filepath: Path | str) -> "RandomForestHAR":
        """Load serialized model from disk."""
        return joblib.load(filepath)


def plot_feature_importance(
    importances: List[Tuple[str, float]],
    top_k: int = 20,
    save_path: Optional[Path | str] = None,
) -> None:
    """Plot horizontal bar chart of top K feature importances."""
    top_features = importances[:top_k]
    names = [f[0] for f in top_features][::-1]
    scores = [f[1] for f in top_features][::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_k), scores, color="#2b5c8f", edgecolor="black", alpha=0.85)
    plt.yticks(range(top_k), names, fontsize=9)
    plt.xlabel("MDI Feature Importance", fontsize=11)
    plt.title(f"Top {top_k} Most Informative IMU Features (Random Forest)", fontsize=13)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info("Saved feature importance plot to %s", save_path)
    plt.close()


def main() -> None:
    """Train and evaluate Random Forest on leak-free subject-wise validation split."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Load unscaled raw windows and labels
    raw_path = repo_root / "data" / "processed" / "X_train_raw.npy"
    if not raw_path.exists():
        from src.data.load_dataset import load_full_split, resolve_dataset_dir
        base_dir = resolve_dataset_dir()
        X_raw, _, _, _ = load_full_split(base_dir, split="train", load_features=False)
        np.save(raw_path, X_raw.astype(np.float32))
    else:
        X_raw = np.load(raw_path)

    y = np.load(repo_root / "data" / "processed" / "y_train.npy")
    subjects = np.load(repo_root / "data" / "processed" / "subject_train.npy")

    logger.info("Total unscaled training tensor shape: %s", X_raw.shape)

    # Leak-free subject-wise split & feature extraction (scaler fitted ONLY on internal train)
    from src.data.splitting import prepare_leak_free_train_val_features

    (
        X_train_feat,
        y_train,
        X_val_feat,
        y_val,
        val_scaler,
        split,
    ) = prepare_leak_free_train_val_features(
        X_raw, y, subjects, val_subjects=[3, 11, 17, 27]
    )

    logger.info("Fitting Random Forest on %d leak-free train samples...", len(X_train_feat))
    rf = RandomForestHAR(n_estimators=150, max_depth=25, random_state=42, n_jobs=1)
    rf.fit(X_train_feat, y_train)
    logger.info("Random Forest training completed in %.2fs", rf.training_time_sec_)

    # Measure inference latency on validation set
    t0 = time.perf_counter()
    y_pred = rf.predict(X_val_feat)
    val_latency_ms = (time.perf_counter() - t0) * 1000.0 / len(y_val)

    val_acc = accuracy_score(y_val, y_pred)
    val_macro_f1 = f1_score(y_val, y_pred, average="macro")
    val_weighted_f1 = f1_score(y_val, y_pred, average="weighted")

    logger.info("=== Random Forest Validation Results (Leak-Free Pipeline) ===")
    logger.info("Validation Accuracy:    %.4f (%.2f%%)", val_acc, val_acc * 100)
    logger.info("Validation Macro F1:    %.4f", val_macro_f1)
    logger.info("Validation Weighted F1: %.4f", val_weighted_f1)
    logger.info("Per-sample Latency:     %.3f ms/window", val_latency_ms)

    print("\nDetailed Classification Report:")
    print(classification_report(y_val, y_pred, target_names=ACTIVITY_NAMES, digits=4))

    # Save model
    model_path = repo_root / "models" / "random_forest.joblib"
    rf.save(model_path)

    # Save feature importances
    importances = rf.get_feature_importances()
    fig_path = repo_root / "reports" / "figures" / "rf_feature_importance.png"
    plot_feature_importance(importances, top_k=20, save_path=fig_path)


if __name__ == "__main__":
    main()
