import torch
import openvino as ov
import sys
import os
from omegaconf import OmegaConf

# Add project src to path to resolve imports
sys.path.append("/Users/mderaznasr/Documents/GitHub/openVINO-project-21/openvino-vla/unifolm-vla/src")

from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model
from single_step_dit_wrapper import SingleStepDiTWrapper

def main():
    print("[INFO] Loading configuration and model...")
    # Load default training config using absolute path
    config_path = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/openvino-vla/unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_train.yaml"
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
    action_horizon = 8     # LIBERO NUM_ACTIONS_CHUNK
    action_dim = 7         # LIBERO ACTION_DIM
    state_dim = 8          # LIBERO PROPRIO_DIM

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
        ov.save_model(ov_model, "export_tests/single_step_dit.xml")
        print("[SUCCESS] Single-step DiT converted and saved to export_tests/single_step_dit.xml")
        
    except Exception as e:
        print(f"[FAILURE] Conversion failed with error:\n{e}")
        # Re-raise to get full traceback if needed
        raise e

if __name__ == "__main__":
    main()
