"""Script to export PyTorch CNN to ONNX and generate optimized variants (FP16, INT8).

Executes:
1. PyTorch to ONNX FP32 export with dynamic batching.
2. Structural validation via onnx.checker.
3. Numerical parity verification on internal validation data.
4. FP16 weight conversion.
5. Static INT8 QDQ quantization using internal validation calibration data.
6. Dynamic INT8 quantization for ConvInteger/MatMulInteger operators.
7. Computational graph operator inspection and reporting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.deployment.onnx_export import export_pytorch_to_onnx, verify_numerical_parity
from src.deployment.quantization import (
    convert_to_fp16,
    inspect_onnx_graph,
    quantize_dynamic_int8,
    quantize_static_int8,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 80)
    logger.info("DAY 3: PYTORCH TO ONNX EXPORT AND OPTIMIZATION PIPELINE")
    logger.info("=" * 80)

    model_pt = repo_root / "models" / "cnn_1d.pt"
    models_opt_dir = repo_root / "models" / "optimized"
    models_opt_dir.mkdir(parents=True, exist_ok=True)

    fp32_onnx = models_opt_dir / "cnn_1d_fp32.onnx"
    fp16_onnx = models_opt_dir / "cnn_1d_fp16.onnx"
    int8_static_onnx = models_opt_dir / "cnn_1d_int8_static.onnx"
    int8_dynamic_onnx = models_opt_dir / "cnn_1d_int8.onnx"

    # Step 1: Export FP32 ONNX
    logger.info("Step 1: Exporting PyTorch CNN to FP32 ONNX...")
    export_pytorch_to_onnx(
        model_path=model_pt,
        output_path=fp32_onnx,
        opset_version=17,
        dynamic_batch=True,
        embed_weights=True,
    )

    # Step 2: Load internal validation samples for parity and calibration
    logger.info("Step 2: Loading internal validation data for parity verification & calibration...")
    raw_train_path = repo_root / "data" / "processed" / "X_train_raw.npy"
    subj_train_path = repo_root / "data" / "processed" / "subject_train.npy"

    if not raw_train_path.exists() or not subj_train_path.exists():
        raise FileNotFoundError(
            "Required data files not found in data/processed/. "
            "Please run 'python -m src.data.preprocess' first."
        )

    X_train_raw = np.load(raw_train_path)
    subj_train = np.load(subj_train_path)

    # Internal validation subjects: [3, 11, 17, 27] (1,401 samples)
    val_mask = np.isin(subj_train, [3, 11, 17, 27])
    internal_train_mask = ~val_mask

    # Load scaler fitted on internal training subjects
    from src.data.preprocess import ChannelWiseStandardScaler
    val_scaler = ChannelWiseStandardScaler()
    val_scaler.fit(X_train_raw[internal_train_mask])
    X_val_scaled = val_scaler.transform(X_train_raw[val_mask])

    # Step 3: Numerical Parity Verification on Validation Data
    logger.info("Step 3: Verifying numerical parity between PyTorch and ONNX Runtime...")
    parity_results = verify_numerical_parity(
        model_pt_path=model_pt,
        model_onnx_path=fp32_onnx,
        test_samples=X_val_scaled[:200],  # 200 representative validation samples
        atol=1e-4,
    )
    print("\n" + "=" * 60)
    print("NUMERICAL PARITY VERIFICATION RESULTS (PyTorch vs ONNX FP32)")
    print("=" * 60)
    print(f"Samples Evaluated        : {parity_results['num_samples']}")
    print(f"Max Absolute Logit Diff : {parity_results['max_absolute_logit_diff']:.6e}")
    print(f"Mean Absolute Logit Diff: {parity_results['mean_absolute_logit_diff']:.6e}")
    print(f"Max Probability Diff     : {parity_results['max_prob_diff']:.6e}")
    print(f"Mean Probability Diff    : {parity_results['mean_prob_diff']:.6e}")
    print(f"Prediction Agreement     : {parity_results['prediction_agreement_pct']:.2f}%")
    print(f"Parity Test Status       : {'PASSED' if parity_results['parity_passed'] else 'FAILED'}")
    print("=" * 60 + "\n")

    # Step 4: FP16 Conversion
    logger.info("Step 4: Converting ONNX model to FP16...")
    convert_to_fp16(fp32_onnx, fp16_onnx)

    # Step 5: Static INT8 QDQ Quantization
    logger.info("Step 5: Performing Static INT8 Quantization using internal calibration data...")
    # Use 200 internal validation windows as calibration dataset
    calib_samples = X_val_scaled[:200]
    quantize_static_int8(fp32_onnx, int8_static_onnx, calibration_data=calib_samples)

    # Step 6: Dynamic INT8 Quantization
    logger.info("Step 6: Performing Dynamic INT8 Quantization...")
    quantize_dynamic_int8(fp32_onnx, int8_dynamic_onnx)

    # Step 7: Inspect graphs
    logger.info("Step 7: Inspecting computational graphs for all exported variants...")
    graph_reports = {}
    for path, name in [
        (fp32_onnx, "onnx_fp32"),
        (fp16_onnx, "onnx_fp16"),
        (int8_static_onnx, "onnx_int8_static"),
        (int8_dynamic_onnx, "onnx_int8_dynamic"),
    ]:
        info = inspect_onnx_graph(path)
        graph_reports[name] = info
        logger.info(
            "[%s] Size: %.2f MB (%d bytes), Precision: %s, Nodes: %d (Quantized: %d, Float: %d)",
            name,
            info["file_size_mb"],
            info["file_size_bytes"],
            info["precision_type"],
            info["total_nodes"],
            info["quantized_nodes"],
            info["floating_point_nodes"],
        )

    # Save export metadata
    meta_path = repo_root / "reports" / "day3_export_summary.json"
    export_summary = {
        "parity_results": parity_results,
        "graph_inspections": graph_reports,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(export_summary, f, indent=2)

    logger.info("Saved export summary to %s", meta_path)
    logger.info("ONNX export and quantization pipeline successfully completed.")


if __name__ == "__main__":
    main()
