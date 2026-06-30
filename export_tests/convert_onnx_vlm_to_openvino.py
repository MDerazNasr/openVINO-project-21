from __future__ import annotations

import argparse
from pathlib import Path

import openvino as ov

REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = REPO_ROOT / "artifacts" / "onnx"
IR_DIR = REPO_ROOT / "artifacts" / "openvino_ir"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert exported Qwen VLM ONNX to OpenVINO IR.")
    parser.add_argument(
        "--onnx-name",
        default="qwen_vlm_backbone.onnx",
        help="Input ONNX filename under artifacts/onnx.",
    )
    parser.add_argument(
        "--output-name",
        default="qwen_vlm_backbone_from_onnx.xml",
        help="Output XML filename under artifacts/openvino_ir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = ONNX_DIR / args.onnx_name
    output_path = IR_DIR / args.output_name
    print(f"[INFO] ONNX path: {onnx_path}")
    print(f"[INFO] ONNX exists: {onnx_path.exists()}")
    if onnx_path.exists():
        print(f"[INFO] ONNX bytes: {onnx_path.stat().st_size}")

    IR_DIR.mkdir(parents=True, exist_ok=True)
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, output_path)
    print(f"[SUCCESS] OpenVINO IR saved to {output_path}")
    print(f"[INFO] XML bytes: {output_path.stat().st_size}")
    bin_path = output_path.with_suffix(".bin")
    print(f"[INFO] BIN exists: {bin_path.exists()}")
    if bin_path.exists():
        print(f"[INFO] BIN bytes: {bin_path.stat().st_size}")


if __name__ == "__main__":
    main()
