"""1D Convolutional Neural Network (CNN) for raw IMU sensor time-series classification.

Operates directly on triaxial accelerometer and gyroscope windows (9 channels, 128 timesteps)
using PyTorch with subject-wise train/validation split, early stopping, and performance profiling.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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


class Conv1DNet(nn.Module):
    """Deep 1D CNN architecture for 9-channel IMU time-series."""

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        dropout_rate: float = 0.35,
    ):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects tensor of shape (batch_size, in_channels, seq_len)."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.gap(x).squeeze(-1)
        logits = self.classifier(x)
        return logits


class CNN1DHAR:
    """Wrapper for 1D CNN training, prediction, evaluation, and serialization."""

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        max_epochs: int = 35,
        patience: int = 8,
        random_seed: int = 42,
        device: Optional[str] = None,
    ):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_seed = random_seed

        torch.manual_seed(random_seed)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = Conv1DNet(
            in_channels=in_channels,
            num_classes=num_classes,
        ).to(self.device)

        self.training_time_sec_: float = 0.0
        self.best_epoch_: int = 0
        self._label_offset: int = 0
        self.is_fitted_: bool = False
        self.history_: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _prepare_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Convert numpy array (N, 128, 9) to torch tensor (N, 9, 128)."""
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        # Permute (N, 128, 9) -> (N, 9, 128)
        X_transposed = np.transpose(X, (0, 2, 1))
        return torch.tensor(X_transposed, dtype=torch.float32)

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "CNN1DHAR":
        """Train CNN with early stopping on validation loss."""
        start_t = time.perf_counter()

        # Handle label indexing: ensure 0-indexed (0..num_classes-1)
        if y_train.min() == 1:
            y_train_zero = y_train - 1
            y_val_zero = y_val - 1
            self._label_offset = 1
        else:
            y_train_zero = y_train.copy()
            y_val_zero = y_val.copy()
            self._label_offset = 0

        t_X_train = self._prepare_tensor(X_train)
        t_y_train = torch.tensor(y_train_zero, dtype=torch.long)
        t_X_val = self._prepare_tensor(X_val)
        t_y_val = torch.tensor(y_val_zero, dtype=torch.long)

        train_dataset = TensorDataset(t_X_train, t_y_train)
        val_dataset = TensorDataset(t_X_val, t_y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val_loss = float("inf")
        best_weights = None
        epochs_no_improve = 0

        logger.info(
            "Starting 1D CNN training for up to %d epochs (device: %s, params: %d)...",
            self.max_epochs,
            self.device,
            self.count_parameters(),
        )

        for epoch in range(1, self.max_epochs + 1):
            # Train step
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_y)

            train_loss = running_loss / len(train_dataset)

            # Validation step
            self.model.eval()
            val_loss_total = 0.0
            correct = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss_total += loss.item() * len(batch_y)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == batch_y).sum().item()

            val_loss = val_loss_total / len(val_dataset)
            val_acc = correct / len(val_dataset)

            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_acc"].append(val_acc)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 5 == 0 or epoch == 1 or epochs_no_improve == 0:
                logger.info(
                    "Epoch [%2d/%2d] - Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f (Best: %.4f @ ep %d)",
                    epoch,
                    self.max_epochs,
                    train_loss,
                    val_loss,
                    val_acc,
                    best_val_loss,
                    self.best_epoch_,
                )

            if epochs_no_improve >= self.patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", self.patience)
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        self.is_fitted_ = True
        self.training_time_sec_ = time.perf_counter() - start_t
        logger.info(
            "1D CNN training complete in %.2fs (Best epoch: %d)",
            self.training_time_sec_,
            self.best_epoch_,
        )
        return self

    def fit_fixed_epochs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
    ) -> "CNN1DHAR":
        """Train CNN for a predetermined, fixed number of epochs without validation data.

        Guarantees zero test-set leakage:
        - Reinitializes the model architecture from scratch using random_seed.
        - Accepts ONLY training arrays (does not accept validation or test arguments).
        - Computes zero validation or test loss during training.
        - Performs zero early stopping, checkpoint selection, or hyperparameter updates
          based on unobserved or test data.
        - Executes exactly `epochs` iterations, derived exclusively from prior validation.

        Args:
            X_train: Training sensor tensor of shape (N, 128, 9).
            y_train: Training labels of shape (N,).
            epochs: Exact number of training epochs to execute.

        Returns:
            self: The fitted CNN1DHAR model.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")

        start_t = time.perf_counter()

        # Reinitialize model from scratch to guarantee no weight carryover
        torch.manual_seed(self.random_seed)
        self.model = Conv1DNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        ).to(self.device)

        # Handle label indexing: ensure 0-indexed (0..num_classes-1)
        if y_train.min() == 1:
            y_train_zero = y_train - 1
            self._label_offset = 1
        else:
            y_train_zero = y_train.copy()
            self._label_offset = 0

        t_X_train = self._prepare_tensor(X_train)
        t_y_train = torch.tensor(y_train_zero, dtype=torch.long)

        train_dataset = TensorDataset(t_X_train, t_y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Initialize history tracking training loss only
        self.history_ = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

        logger.info(
            "Starting fixed-epoch 1D CNN training for %d epochs (device: %s, params: %d)...",
            epochs,
            self.device,
            self.count_parameters(),
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_y)

            train_loss = running_loss / len(train_dataset)
            self.history_["train_loss"].append(train_loss)

            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                logger.info(
                    "Epoch [%2d/%2d] - Train Loss: %.4f (Fixed schedule, zero test/val leakage)",
                    epoch,
                    epochs,
                    train_loss,
                )

        self.best_epoch_ = epochs
        self.is_fitted_ = True
        self.training_time_sec_ = time.perf_counter() - start_t
        logger.info(
            "Fixed-epoch 1D CNN training complete in %.2fs (%d epochs executed)",
            self.training_time_sec_,
            epochs,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict activity labels (1..6)."""
        if not self.is_fitted_:
            raise RuntimeError("CNN1DHAR has not been trained yet. Call fit() or load() first.")
        self.model.eval()
        t_X = self._prepare_tensor(X).to(self.device)
        loader = DataLoader(TensorDataset(t_X), batch_size=self.batch_size, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                logits = self.model(batch_x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
        flat_preds = np.concatenate(all_preds, axis=0)
        return flat_preds + getattr(self, "_label_offset", 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("CNN1DHAR has not been trained yet. Call fit() or load() first.")
        self.model.eval()
        t_X = self._prepare_tensor(X).to(self.device)
        loader = DataLoader(TensorDataset(t_X), batch_size=self.batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def save(self, filepath: Path | str) -> None:
        """Save model checkpoint to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "training_time_sec": self.training_time_sec_,
            "best_epoch": self.best_epoch_,
            "label_offset": self._label_offset,
            "history": self.history_,
        }
        torch.save(checkpoint, filepath)
        logger.info("Saved 1D CNN checkpoint to %s", filepath)

    @classmethod
    def load(cls, filepath: Path | str, device: Optional[str] = None) -> "CNN1DHAR":
        """Load model checkpoint from disk."""
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        instance = cls(
            in_channels=checkpoint["in_channels"],
            num_classes=checkpoint["num_classes"],
            device=device,
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.training_time_sec_ = checkpoint.get("training_time_sec", 0.0)
        instance.best_epoch_ = checkpoint.get("best_epoch", 0)
        instance._label_offset = checkpoint.get("label_offset", 0)
        instance.history_ = checkpoint.get("history", {})
        instance.is_fitted_ = True
        return instance


def main() -> None:
    """Train and evaluate 1D CNN on leak-free subject-wise validation split."""
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

    # Leak-free subject-wise split & scaling (scaler fitted ONLY on internal train)
    from src.data.splitting import prepare_leak_free_train_val_tensors

    (
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        val_scaler,
        split,
    ) = prepare_leak_free_train_val_tensors(
        X_raw, y, subjects, val_subjects=[3, 11, 17, 27]
    )

    logger.info("Fitting 1D CNN on %d leak-free train samples...", len(X_train_scaled))
    cnn = CNN1DHAR(
        in_channels=9,
        num_classes=6,
        batch_size=64,
        lr=0.001,
        max_epochs=35,
        patience=8,
        random_seed=42,
    )
    cnn.fit(X_train_scaled, y_train, X_val_scaled, y_val)

    # Measure inference latency on validation set
    t0 = time.perf_counter()
    y_pred = cnn.predict(X_val_scaled)
    val_latency_ms = (time.perf_counter() - t0) * 1000.0 / len(y_val)

    val_acc = accuracy_score(y_val, y_pred)
    val_macro_f1 = f1_score(y_val, y_pred, average="macro")
    val_weighted_f1 = f1_score(y_val, y_pred, average="weighted")

    logger.info("=== 1D CNN Validation Results (Leak-Free Pipeline) ===")
    logger.info("Validation Accuracy:    %.4f (%.2f%%)", val_acc, val_acc * 100)
    logger.info("Validation Macro F1:    %.4f", val_macro_f1)
    logger.info("Validation Weighted F1: %.4f", val_weighted_f1)
    logger.info("Per-sample Latency:     %.3f ms/window", val_latency_ms)

    print("\nDetailed Classification Report:")
    print(classification_report(y_val, y_pred, target_names=ACTIVITY_NAMES, digits=4))

    # Save model
    model_path = repo_root / "models" / "cnn_1d.pt"
    cnn.save(model_path)


if __name__ == "__main__":
    main()
