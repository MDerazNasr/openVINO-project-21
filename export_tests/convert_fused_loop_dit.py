import torch
import numpy as np
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
from unifolm_vla.model.modules.action_model.DiT_ActionHeader_v2 import FlowmatchingActionHead_v2
from FullLoopDiTWrapper import FullLoopDiTWrapper

def main():
    print("[INFO] Loading configuration and model...")
    # Seed for deterministic export
    torch.manual_seed(42)
    
    config_path = UNIFOLM_SRC / "unifolm_vla" / "config" / "training" / "unifolm_vla_train.yaml"
    config = OmegaConf.load(config_path)
    
    # Use v2 natively patched model
    action_model = FlowmatchingActionHead_v2(full_config=config)
    
    # Wrap with unrolled loop
    num_steps = 4
    wrapper = FullLoopDiTWrapper(action_model, num_steps=num_steps).eval()

    print(f"[INFO] Creating inputs for {num_steps}-step fused loop...")
    batch_size = 1
    seq_len = 512          
    vl_dim = 2048          
    action_horizon = 8     
    action_dim = 7         
    state_dim = 8          

    vl_embs = torch.randn(batch_size, seq_len, vl_dim, dtype=torch.float32)
    noise = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32)
    state = torch.randn(batch_size, 1, state_dim, dtype=torch.float32)

    print("[INFO] Converting fused loop model to OpenVINO IR (This may take a while)...")
    try:
        ov_model = ov.convert_model(
            wrapper,
            example_input=(vl_embs, noise, state),
        )

        os.makedirs(IR_DIR, exist_ok=True)
        ov.save_model(ov_model, IR_DIR / "fused_loop_dit.xml")
        print(f"[SUCCESS] Fused Loop DiT converted and saved to {IR_DIR / 'fused_loop_dit.xml'}")
        
    except Exception as e:
        print(f"[FAILURE] Conversion failed with error:\n{e}")

if __name__ == "__main__":
    main()
