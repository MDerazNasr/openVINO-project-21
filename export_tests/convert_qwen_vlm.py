from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import openvino as ov
import sys
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIFOLM_SRC = REPO_ROOT / "openvino-vla" / "unifolm-vla" / "src"
IR_DIR = REPO_ROOT / "artifacts" / "openvino_ir"

# Add project src to path
sys.path.append(str(UNIFOLM_SRC))
from unifolm_vla.model.modules.vlm.QWen2_5 import get_qwen2_5_interface

class QwenVLForwardWrapper(nn.Module):
    """
    Wraps the Qwen2.5-VL model to export only the feature extraction path (last_hidden_states).
    """
    def __init__(self, vlm_interface):
        super().__init__()
        self.model = vlm_interface.model

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        # We target the last_hidden_state which is used by the DiT action head
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(hidden_states=[torch.randn(1, 512, 2048)])


class MockInterface:
    def __init__(self):
        self.model = MockModel()


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Qwen2.5-VL feature extractor to OpenVINO IR.")
    parser.add_argument(
        "--base-vlm",
        default=os.environ.get("UNIFOLM_BASE_VLM"),
        help="HuggingFace model id or local path for the real Qwen/UnifoLM VLM checkpoint.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("UNIFOLM_ATTN_IMPLEMENTATION", "eager"),
        help="Attention implementation to request from transformers. Use eager for CPU/export-friendly conversion.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=os.environ.get("UNIFOLM_TORCH_DTYPE", "float32"),
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for loading the VLM before OpenVINO conversion.",
    )
    parser.add_argument(
        "--device-map",
        default=os.environ.get("UNIFOLM_DEVICE_MAP", ""),
        help="Optional transformers device_map. Leave empty for normal CPU loading on the Windows runner.",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow fallback to a tiny dummy model for structural export testing. Do not use for latency claims.",
    )
    parser.add_argument(
        "--output-name",
        default="qwen_vlm_backbone.xml",
        help="Output XML filename under artifacts/openvino_ir.",
    )
    return parser.parse_args()


def load_qwen_vlm_interface(base_vlm, attn_implementation="eager", torch_dtype="float32", device_map=""):
    config_path = UNIFOLM_SRC / "unifolm_vla" / "config" / "training" / "unifolm_vla_train.yaml"
    config = OmegaConf.load(config_path)
    if base_vlm:
        config.framework.qwenvl.base_vlm = base_vlm
    config.framework.qwenvl.attn_implementation = attn_implementation
    config.framework.qwenvl.torch_dtype = torch_dtype
    config.framework.qwenvl.device_map = device_map
    print(f"[INFO] base_vlm: {config.framework.qwenvl.base_vlm}")
    print(f"[INFO] attn_implementation: {config.framework.qwenvl.attn_implementation}")
    print(f"[INFO] torch_dtype: {config.framework.qwenvl.torch_dtype}")
    print(f"[INFO] device_map: {config.framework.qwenvl.device_map or '<none>'}")
    return get_qwen2_5_interface(config=config)


def build_qwen_export_inputs(vlm_interface):
    image = Image.new("RGB", (224, 224), color=(127, 127, 127))
    inputs = vlm_interface.build_qwenvl_inputs([[image]], ["move the robot arm to the target"])
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"].to(torch.float32)
    image_grid_thw = inputs["image_grid_thw"]
    print(f"[INFO] input_ids shape: {tuple(input_ids.shape)}")
    print(f"[INFO] attention_mask shape: {tuple(attention_mask.shape)}")
    print(f"[INFO] pixel_values shape: {tuple(pixel_values.shape)}")
    print(f"[INFO] image_grid_thw shape: {tuple(image_grid_thw.shape)}")
    return input_ids, attention_mask, pixel_values, image_grid_thw


def main():
    args = parse_args()
    print("[INFO] Loading configuration and Qwen model...")
    torch.manual_seed(42)

    using_mock = False
    try:
        vlm_interface = load_qwen_vlm_interface(
            base_vlm=args.base_vlm,
            attn_implementation=args.attn_implementation,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        wrapper = QwenVLForwardWrapper(vlm_interface).eval()
    except Exception as e:
        if not args.allow_mock:
            raise RuntimeError(
                "Could not load the real Qwen2.5-VL interface. "
                "This script no longer falls back to a mock model by default because mock exports are invalid for VLM/full-VLA latency claims. "
                "Use --allow-mock only for structural conversion tests."
            ) from e
        print(f"[WARNING] Could not load full model: {e}")
        print("[WARNING] Falling back to a dummy architecture. This output is not valid for latency benchmarking.")
        wrapper = QwenVLForwardWrapper(MockInterface()).eval()
        using_mock = True

    print("[INFO] Creating dummy multimodal inputs...")
    if using_mock:
        batch_size = 1
        seq_len = 512
        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        pixel_values = torch.randn((1, 3, 224, 224), dtype=torch.float32)
        image_grid_thw = torch.tensor([[1, 28, 28]], dtype=torch.long)
    else:
        input_ids, attention_mask, pixel_values, image_grid_thw = build_qwen_export_inputs(vlm_interface)

    print("[INFO] Converting VLM backbone to OpenVINO IR...")
    try:
        ov_model = ov.convert_model(
            wrapper,
            example_input=(input_ids, attention_mask, pixel_values, image_grid_thw),
        )

        os.makedirs(IR_DIR, exist_ok=True)
        output_path = IR_DIR / args.output_name
        ov.save_model(ov_model, output_path)
        print(f"[SUCCESS] Qwen VLM converted and saved to {output_path}.")
        
    except Exception as e:
        print(f"[FAILURE] VLM Conversion failed with error:\n{e}")
        raise

if __name__ == "__main__":
    main()
