"""Unit tests for dataset loading functions."""

from pathlib import Path
import numpy as np
import pytest

from src.data.load_dataset import (
    CHANNEL_NAMES,
    load_activity_labels,
    load_feature_names,
    load_labels,
    load_subjects,
    load_inertial_signals,
    resolve_dataset_dir,
)


@pytest.fixture(scope="module")
def dataset_path() -> Path:
    """Fixture providing path to the extracted UCI HAR dataset."""
    return resolve_dataset_dir()


def test_load_activity_labels(dataset_path: Path):
    """Test loading activity labels returns 6 activities with expected names."""
    mapping = load_activity_labels(dataset_path)
    assert len(mapping) == 6
    assert mapping[1] == "WALKING"
    assert mapping[2] == "WALKING_UPSTAIRS"
    assert mapping[3] == "WALKING_DOWNSTAIRS"
    assert mapping[4] == "SITTING"
    assert mapping[5] == "STANDING"
    assert mapping[6] == "LAYING"


def test_load_feature_names(dataset_path: Path):
    """Test loading feature names returns 561 items."""
    features = load_feature_names(dataset_path)
    assert len(features) == 561
    assert "tBodyAcc-mean()-X" in features[0]


def test_load_labels_and_subjects_lengths(dataset_path: Path):
    """Test train and test label and subject arrays match expected sample counts."""
    y_train = load_labels(dataset_path, split="train")
    subj_train = load_subjects(dataset_path, split="train")
    assert len(y_train) == 7352
    assert len(subj_train) == 7352

    y_test = load_labels(dataset_path, split="test")
    subj_test = load_subjects(dataset_path, split="test")
    assert len(y_test) == 2947
    assert len(subj_test) == 2947


def test_load_inertial_signals_shapes(dataset_path: Path):
    """Test loading inertial signal tensors yields exact (N, 128, 9) shapes."""
    X_train = load_inertial_signals(dataset_path, split="train")
    assert X_train.shape == (7352, 128, 9)
    assert np.issubdtype(X_train.dtype, np.floating)

    X_test = load_inertial_signals(dataset_path, split="test")
    assert X_test.shape == (2947, 128, 9)
    assert np.issubdtype(X_test.dtype, np.floating)


def test_load_missing_file_raises(tmp_path: Path):
    """Test that loading from invalid directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_activity_labels(tmp_path)
