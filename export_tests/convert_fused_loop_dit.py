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
from unifolm_vla.rlds_dataloader.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from FullLoopDiTWrapper import FullLoopDiTWrapper

def main():
    print("[INFO] Loading configuration and model...")
    # Seed for deterministic export
    torch.manual_seed(42)
    
    config_path = UNIFOLM_SRC / "unifolm_vla" / "config" / "training" / "unifolm_vla_train.yaml"
    config = OmegaConf.load(config_path)
    seq_len = int(os.environ.get("VLA_VL_SEQ_LEN", "512"))
    vl_dim = int(os.environ.get("VLA_VL_DIM", str(config.framework.action_model.diffusion_model_cfg.cross_attention_dim)))
    config.framework.action_model.vl_hidden_dim = vl_dim
    config.framework.action_model.diffusion_model_cfg.cross_attention_dim = vl_dim
    print(f"[INFO] VLM conditioning shape: seq_len={seq_len}, hidden_dim={vl_dim}")
    
    # Use v2 natively patched model
    action_model = FlowmatchingActionHead_v2(full_config=config)
    
    # Wrap with unrolled loop
    num_steps = 4
    wrapper = FullLoopDiTWrapper(action_model, num_steps=num_steps).eval()

    print(f"[INFO] Creating inputs for {num_steps}-step fused loop...")
    batch_size = 1
    action_horizon = NUM_ACTIONS_CHUNK
    action_dim = ACTION_DIM
    state_dim = PROPRIO_DIM

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
        raise

if __name__ == "__main__":
    main()
