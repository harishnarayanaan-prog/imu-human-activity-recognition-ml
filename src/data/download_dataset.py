"""UCI Human Activity Recognition (HAR) Dataset Downloader and Extractor.

This module automates the acquisition and extraction of the official UCI HAR dataset
from the official UCI Machine Learning Repository.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OFFICIAL_UCI_URL = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
)
LEGACY_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
)

EXPECTED_CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def check_archive_validity(archive_path: Path) -> bool:
    """Check if the downloaded zip archive exists and is not corrupted."""
    if not archive_path.is_file():
        return False
    if archive_path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            first_bad = zf.testzip()
            if first_bad is not None:
                logger.warning("Corrupt file detected in archive: %s", first_bad)
                return False
        return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        logger.warning("Invalid zip archive %s: %s", archive_path, e)
        return False


def download_dataset(
    url: str = OFFICIAL_UCI_URL,
    target_dir: Path | str = "data/raw",
    archive_name: str = "uci_har_dataset.zip",
    chunk_size: int = 1024 * 1024,
    timeout: int = 60,
) -> Path:
    """Download the dataset from the official URL if not already present and valid."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    archive_path = target_path / archive_name

    if check_archive_validity(archive_path):
        logger.info("Valid archive already exists at: %s. Skipping download.", archive_path)
        return archive_path

    logger.info("Initiating download from official source: %s", url)
    headers = {"User-Agent": "IMU-HAR-ML-Research/1.0 (Windows NT 10.0; Win64; x64)"}

    # Stream download to avoid loading entire archive into memory
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.warning("Primary download failed (%s). Attempting legacy UCI URL...", exc)
        response = requests.get(LEGACY_UCI_URL, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))
    temp_archive = archive_path.with_suffix(".tmp")
    downloaded_bytes = 0

    try:
        with open(temp_archive, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    if total_bytes > 0:
                        percent = (downloaded_bytes / total_bytes) * 100
                        logger.info("Downloaded %d / %d bytes (%.1f%%)", downloaded_bytes, total_bytes, percent)
                    else:
                        logger.info("Downloaded %d bytes...", downloaded_bytes)

        # Rename temp to target
        if temp_archive.exists():
            if archive_path.exists():
                archive_path.unlink()
            temp_archive.rename(archive_path)

        logger.info("Download completed successfully: %s (%d bytes)", archive_path, archive_path.stat().st_size)
    finally:
        if temp_archive.exists():
            temp_archive.unlink(missing_ok=True)

    if not check_archive_validity(archive_path):
        raise RuntimeError(f"Downloaded archive at {archive_path} is invalid or corrupted.")

    return archive_path


def extract_dataset(
    archive_path: Path | str,
    extracted_dir: Path | str = "data/extracted",
) -> Path:
    """Extract the archive into the extracted directory.

    Handles both direct archives and nested zip archives (e.g., 'UCI HAR Dataset.zip' inside).
    """
    archive_path = Path(archive_path)
    extracted_path = Path(extracted_dir)
    extracted_path.mkdir(parents=True, exist_ok=True)

    target_har_dir = extracted_path / "UCI HAR Dataset"
    if target_har_dir.is_dir() and (target_har_dir / "train").is_dir():
        logger.info("Extracted dataset directory already exists at: %s", target_har_dir)
        return target_har_dir

    logger.info("Extracting archive %s to %s...", archive_path, extracted_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extracted_path)

    # Check if extraction yielded a nested zip (e.g. 'UCI HAR Dataset.zip')
    nested_zip = extracted_path / "UCI HAR Dataset.zip"
    if nested_zip.is_file():
        logger.info("Detected nested archive %s. Extracting...", nested_zip)
        with zipfile.ZipFile(nested_zip, "r") as nzf:
            nzf.extractall(extracted_path)
        nested_zip.unlink()

    # Search for UCI HAR Dataset folder
    if not target_har_dir.is_dir():
        # Look for subfolder
        candidates = list(extracted_path.glob("**/UCI HAR Dataset"))
        if candidates and candidates[0].is_dir():
            actual_dir = candidates[0]
            if actual_dir != target_har_dir:
                shutil.move(str(actual_dir), str(target_har_dir))

    if not target_har_dir.is_dir():
        raise RuntimeError(f"Expected extracted directory 'UCI HAR Dataset' not found in {extracted_path}")

    logger.info("Dataset successfully extracted to: %s", target_har_dir)
    return target_har_dir


def verify_extracted_dataset(dataset_dir: Path | str) -> Dict[str, int]:
    """Verify that all expected dataset files exist and have non-zero file sizes."""
    base = Path(dataset_dir)
    required_files = [
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
        required_files.append(base / "train" / "Inertial Signals" / f"{ch}_train.txt")
        required_files.append(base / "test" / "Inertial Signals" / f"{ch}_test.txt")

    file_sizes = {}
    missing_files = []
    zero_size_files = []

    for fpath in required_files:
        if not fpath.is_file():
            missing_files.append(str(fpath))
        else:
            sz = fpath.stat().st_size
            if sz == 0:
                zero_size_files.append(str(fpath))
            file_sizes[str(fpath.relative_to(base))] = sz

    if missing_files:
        raise FileNotFoundError(f"Missing required dataset files ({len(missing_files)}): {missing_files[:5]}")
    if zero_size_files:
        raise ValueError(f"Found empty files ({len(zero_size_files)}): {zero_size_files[:5]}")

    logger.info("All %d required files verified with positive file sizes.", len(required_files))
    return file_sizes


def main():
    """Main CLI execution flow."""
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    extracted_dir = root / "data" / "extracted"

    logger.info("Executing dataset acquisition pipeline...")
    archive_path = download_dataset(target_dir=raw_dir)
    har_dir = extract_dataset(archive_path, extracted_dir=extracted_dir)
    sizes = verify_extracted_dataset(har_dir)
    logger.info("Dataset download, extraction, and file verification completed successfully.")


if __name__ == "__main__":
    main()
