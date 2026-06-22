from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import openvino as ov
import sys
import os
from pathlib import Path
from omegaconf import OmegaConf

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


def main():
    args = parse_args()
    print("[INFO] Loading configuration and Qwen model...")
    torch.manual_seed(42)
    
    config_path = UNIFOLM_SRC / "unifolm_vla" / "config" / "training" / "unifolm_vla_train.yaml"
    config = OmegaConf.load(config_path)
    
    try:
        vlm_interface = get_qwen2_5_interface(config=config)
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

    print("[INFO] Creating dummy multimodal inputs...")
    batch_size = 1
    seq_len = 512
    
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    
    # Qwen-VL pixel values and grid thw
    # Typically [num_patches, hidden_dim] after vision encoder, but here we mock raw-ish inputs
    pixel_values = torch.randn((1, 3, 224, 224), dtype=torch.float32)
    image_grid_thw = torch.tensor([[1, 28, 28]], dtype=torch.long)

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
