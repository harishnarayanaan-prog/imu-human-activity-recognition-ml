"""Metadata generation and verification utility for processed IMU datasets.

Audits processed data files in `data/processed/`, computes file hashes,
verifies data integrity, records tensor dimensions, and writes `data_dictionary.md`
and metadata summaries.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def audit_processed_directory(processed_dir: Path | str = "data/processed") -> Dict:
    """Audit all files in the processed directory and extract data specifications."""
    pdir = Path(processed_dir)
    if not pdir.is_dir():
        raise FileNotFoundError(f"Processed directory not found: {pdir}")

    required_artifacts = [
        "X_train.npy",
        "y_train.npy",
        "subject_train.npy",
        "X_test.npy",
        "y_test.npy",
        "subject_test.npy",
        "label_mapping.json",
        "preprocessing_metadata.json",
    ]

    audit_records = {}
    missing = []
    for art in required_artifacts:
        art_path = pdir / art
        if not art_path.is_file():
            missing.append(art)
            continue

        stat = art_path.stat()
        record = {
            "file_name": art,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 3),
            "sha256": compute_sha256(art_path),
        }

        if art.endswith(".npy"):
            arr = np.load(art_path)
            record["shape"] = list(arr.shape)
            record["dtype"] = str(arr.dtype)
            record["nan_count"] = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0

        audit_records[art] = record

    if missing:
        raise FileNotFoundError(f"Missing expected processed artifacts: {missing}")

    return audit_records


def main():
    """CLI execution flow."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    processed_dir = ROOT_DIR / "data" / "processed"
    audit = audit_processed_directory(processed_dir)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
