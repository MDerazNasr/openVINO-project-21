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

# Add project src to path to resolve imports
sys.path.append(str(UNIFOLM_SRC))

from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model
from unifolm_vla.rlds_dataloader.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from single_step_dit_wrapper import SingleStepDiTWrapper

def main():
    print("[INFO] Enforcing strict determinism...")
    torch.manual_seed(42)
    np.random.seed(42)

    print("[INFO] Loading configuration and model...")
    # Load default training config.
    config_path = UNIFOLM_SRC / "unifolm_vla" / "config" / "training" / "unifolm_vla_train.yaml"
    config = OmegaConf.load(config_path)
    
    # Instantiate the action model (DiT Flowmatching head)
    # We use config to avoid loading huge weights for this test
    action_model = get_action_model(config=config)
    
    # Wrap with our single-step instrumentation fix
    wrapper = SingleStepDiTWrapper(action_model).eval()

    print("[INFO] Creating dummy inputs (using traced LIBERO dimensions)...")
    batch_size = 1
    seq_len = 512          # Qwen output seq_len
    vl_dim = 2048          # Qwen hidden dimension (vl_hidden_dim)
    action_horizon = NUM_ACTIONS_CHUNK
    action_dim = ACTION_DIM
    state_dim = PROPRIO_DIM

    vl_embs = torch.randn(batch_size, seq_len, vl_dim, dtype=torch.float32)
    actions = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32)
    state = torch.randn(batch_size, 1, state_dim, dtype=torch.float32)
    timesteps = torch.zeros(batch_size, dtype=torch.long)

    print("[INFO] Converting model to OpenVINO IR...")
    try:
        ov_model = ov.convert_model(
            wrapper,
            example_input=(vl_embs, actions, state, timesteps),
        )

        print("[INFO] Saving IR...")
    
        os.makedirs(IR_DIR, exist_ok=True)
        ov.save_model(ov_model, IR_DIR / "single_step_dit.xml")
        print(f"[SUCCESS] Single-step DiT converted and saved to {IR_DIR / 'single_step_dit.xml'}")
        
    except Exception as e:
        print(f"[FAILURE] Conversion failed with error:\n{e}")
        # Re-raise to get full traceback if needed
        raise e

if __name__ == "__main__":
    main()
