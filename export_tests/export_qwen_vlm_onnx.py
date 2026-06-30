from __future__ import annotations

import argparse
from pathlib import Path

import torch

from convert_qwen_vlm import QwenVLForwardWrapper, build_qwen_export_inputs, load_qwen_vlm_interface

REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = REPO_ROOT / "artifacts" / "onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Export Qwen2.5-VL feature extractor to ONNX.")
    parser.add_argument(
        "--base-vlm",
        required=True,
        help="HuggingFace model id or local path for the real Qwen/UnifoLM VLM checkpoint.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Attention implementation to request from transformers.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for loading the VLM before ONNX export.",
    )
    parser.add_argument(
        "--device-map",
        default="",
        help="Optional transformers device_map. Leave empty for CPU loading on the Windows runner.",
    )
    parser.add_argument(
        "--output-name",
        default="qwen_vlm_backbone.onnx",
        help="Output ONNX filename under artifacts/onnx.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("[INFO] Loading real Qwen VLM interface for ONNX export...")
    vlm_interface = load_qwen_vlm_interface(
        base_vlm=args.base_vlm,
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    wrapper = QwenVLForwardWrapper(vlm_interface).eval()

    print("[INFO] Building processor-backed multimodal example inputs...")
    input_ids, attention_mask, pixel_values, image_grid_thw = build_qwen_export_inputs(vlm_interface)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ONNX_DIR / args.output_name
    print(f"[INFO] Exporting ONNX to {output_path}")

    common_kwargs = dict(
        args=(input_ids, attention_mask, pixel_values, image_grid_thw),
        f=str(output_path),
        input_names=["input_ids", "attention_mask", "pixel_values", "image_grid_thw"],
        output_names=["last_hidden_state"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    try:
        torch.onnx.export(wrapper, **common_kwargs, external_data=True)
    except TypeError:
        torch.onnx.export(wrapper, **common_kwargs)

    print(f"[SUCCESS] ONNX export saved to {output_path}")


if __name__ == "__main__":
    main()
