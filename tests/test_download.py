"""Unit tests for dataset download and archive validation logic."""

import zipfile
from pathlib import Path
import pytest

from src.data.download_dataset import (
    check_archive_validity,
    verify_extracted_dataset,
    EXPECTED_CHANNELS,
)


def test_archive_validity_with_nonexistent_file(tmp_path: Path):
    """Test check_archive_validity returns False for nonexistent paths."""
    fake_zip = tmp_path / "nonexistent.zip"
    assert check_archive_validity(fake_zip) is False


def test_archive_validity_with_empty_file(tmp_path: Path):
    """Test check_archive_validity returns False for zero-byte files."""
    empty_zip = tmp_path / "empty.zip"
    empty_zip.touch()
    assert check_archive_validity(empty_zip) is False


def test_archive_validity_with_corrupt_file(tmp_path: Path):
    """Test check_archive_validity returns False for corrupt archives."""
    corrupt_zip = tmp_path / "corrupt.zip"
    corrupt_zip.write_bytes(b"PK\x03\x04this is not a real zip file content")
    assert check_archive_validity(corrupt_zip) is False


def test_archive_validity_with_valid_zip(tmp_path: Path):
    """Test check_archive_validity returns True for well-formed zip files."""
    valid_zip = tmp_path / "valid.zip"
    with zipfile.ZipFile(valid_zip, "w") as zf:
        zf.writestr("test.txt", "hello world")
    assert check_archive_validity(valid_zip) is True


def test_verify_extracted_dataset_missing_file_raises(tmp_path: Path):
    """Test verify_extracted_dataset raises FileNotFoundError on missing items."""
    with pytest.raises(FileNotFoundError):
        verify_extracted_dataset(tmp_path)


def test_verify_extracted_dataset_empty_file_raises(tmp_path: Path):
    """Test verify_extracted_dataset raises ValueError when a file has 0 bytes."""
    base = tmp_path / "dataset"
    (base / "train" / "Inertial Signals").mkdir(parents=True)
    (base / "test" / "Inertial Signals").mkdir(parents=True)

    required = [
        base / "features.txt",
        base / "activity_labels.txt",
        base / "train" / "X_train.txt",
        base / "train" / "y_train.txt",
        base / "train" / "subject_train.txt",
        base / "test" / "X_test.txt",
        base / "test" / "y_test.txt",
        base / "test" / "subject_test.txt",
    ]
    for ch in EXPECTED_CHANNELS:
        required.append(base / "train" / "Inertial Signals" / f"{ch}_train.txt")
        required.append(base / "test" / "Inertial Signals" / f"{ch}_test.txt")

    for f in required:
        f.touch()  # Creates 0-byte file

    with pytest.raises(ValueError, match="Found empty files"):
        verify_extracted_dataset(base)
