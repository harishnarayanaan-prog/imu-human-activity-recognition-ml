"""Support Vector Machine (SVM) Classifier for IMU Human Activity Recognition.

Uses a scikit-learn Pipeline with StandardScaler fitted strictly on training data
and an RBF kernel SVC with probability estimates enabled.
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
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.data.splitting import subject_wise_train_val_split

logger = logging.getLogger(__name__)

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


class SVMHAR:
    """Support Vector Machine pipeline for HAR classification."""

    def __init__(
        self,
        C: float = 10.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        probability: bool = True,
        random_state: int = 42,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        C=self.C,
                        kernel=self.kernel,
                        gamma=self.gamma,
                        probability=self.probability,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.training_time_sec_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMHAR":
        """Fit scaler and SVM strictly on provided training data."""
        start_t = time.perf_counter()
        self.pipeline.fit(X, y)
        self.training_time_sec_ = time.perf_counter() - start_t
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict activity labels (transformed via pipeline scaler)."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.pipeline.predict_proba(X)

    def save(self, filepath: Path | str) -> None:
        """Serialize model to disk using joblib."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved SVMHAR to %s", filepath)

    @classmethod
    def load(cls, filepath: Path | str) -> "SVMHAR":
        """Load serialized model from disk."""
        return joblib.load(filepath)


def main() -> None:
    """Train and evaluate SVM on subject-wise validation split."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Load engineered features and labels
    X_feat = np.load(repo_root / "data" / "processed" / "X_train_features.npy")
    y = np.load(repo_root / "data" / "processed" / "y_train.npy")
    subjects = np.load(repo_root / "data" / "processed" / "subject_train.npy")

    # Subject-wise train/val split
    split = subject_wise_train_val_split(
        X_feat, y, subjects, val_subjects=[3, 11, 17, 27]
    )

    logger.info("Fitting SVM (RBF kernel, C=10.0) on %d train samples...", len(split.X_train))
    svm = SVMHAR(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
    svm.fit(split.X_train, split.y_train)
    logger.info("SVM training completed in %.2fs", svm.training_time_sec_)

    # Measure inference latency on validation set
    t0 = time.perf_counter()
    y_pred = svm.predict(split.X_val)
    val_latency_ms = (time.perf_counter() - t0) * 1000.0 / len(split.X_val)

    val_acc = accuracy_score(split.y_val, y_pred)
    val_macro_f1 = f1_score(split.y_val, y_pred, average="macro")
    val_weighted_f1 = f1_score(split.y_val, y_pred, average="weighted")

    logger.info("=== SVM Validation Results ===")
    logger.info("Validation Accuracy:    %.4f (%.2f%%)", val_acc, val_acc * 100)
    logger.info("Validation Macro F1:    %.4f", val_macro_f1)
    logger.info("Validation Weighted F1: %.4f", val_weighted_f1)
    logger.info("Per-sample Latency:     %.3f ms/window", val_latency_ms)

    print("\nDetailed Classification Report:")
    print(classification_report(split.y_val, y_pred, target_names=ACTIVITY_NAMES, digits=4))

    # Save model
    model_path = repo_root / "models" / "svm.joblib"
    svm.save(model_path)


if __name__ == "__main__":
    main()
