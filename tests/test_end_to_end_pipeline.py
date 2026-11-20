"""End-to-end integration test validating the real downloaded and processed dataset."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.data.load_dataset import resolve_dataset_dir


@pytest.fixture(scope="module")
def processed_dir() -> Path:
    """Fixture providing path to processed directory."""
    root = Path(__file__).resolve().parents[1]
    pdir = root / "data" / "processed"
    if not pdir.is_dir():
        pytest.skip("Processed dataset not yet generated.")
    return pdir


def test_processed_files_exist(processed_dir: Path):
    """Verify all required processed artifacts are present."""
    required = [
        "X_train.npy",
        "y_train.npy",
        "subject_train.npy",
        "X_test.npy",
        "y_test.npy",
        "subject_test.npy",
        "label_mapping.json",
        "preprocessing_metadata.json",
    ]
    for r in required:
        assert (processed_dir / r).is_file(), f"Missing processed file: {r}"


def test_processed_tensor_shapes_and_types(processed_dir: Path):
    """Verify exact tensor shapes and zero missing values in processed outputs."""
    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    subj_train = np.load(processed_dir / "subject_train.npy")

    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    subj_test = np.load(processed_dir / "subject_test.npy")

    # Shapes
    assert X_train.shape == (7352, 128, 9)
    assert y_train.shape == (7352,)
    assert subj_train.shape == (7352,)

    assert X_test.shape == (2947, 128, 9)
    assert y_test.shape == (2947,)
    assert subj_test.shape == (2947,)

    # Data types
    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32
    assert np.issubdtype(y_train.dtype, np.integer)
    assert np.issubdtype(y_test.dtype, np.integer)

    # Missing values
    assert np.isnan(X_train).sum() == 0
    assert np.isnan(X_test).sum() == 0
    assert np.isinf(X_train).sum() == 0
    assert np.isinf(X_test).sum() == 0

    # Labels 0..5
    assert set(y_train.tolist()) == {0, 1, 2, 3, 4, 5}
    assert set(y_test.tolist()) == {0, 1, 2, 3, 4, 5}

    # Subject disjointness
    tr_subjs = set(subj_train.tolist())
    te_subjs = set(subj_test.tolist())
    assert len(tr_subjs.intersection(te_subjs)) == 0
    assert len(tr_subjs.union(te_subjs)) == 30


def test_metadata_and_label_mapping_validity(processed_dir: Path):
    """Verify metadata json files parse cleanly and contain expected keys."""
    with open(processed_dir / "label_mapping.json", "r", encoding="utf-8") as f:
        lbl_map = json.load(f)
    assert "zero_index_to_activity" in lbl_map
    assert len(lbl_map["zero_index_to_activity"]) == 6

    with open(processed_dir / "preprocessing_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["train_samples"] == 7352
    assert meta["test_samples"] == 2947
    assert meta["subject_leakage_verified"] is True
