import torch
import numpy as np
from omegaconf import OmegaConf
from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model

def trace():
    # Load config
    config = OmegaConf.load("openvino-vla/unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_train.yaml")
    
    # Instantiate action model directly to avoid loading huge VLM weights
    action_model = get_action_model(config=config)
    action_model.eval()
    
    # Mock VLM output: [B, L, H]
    # From config, cross_attention_dim is 2048.
    B = 1
    L = 512
    H = 2048
    vl_embs = torch.randn((B, L, H), dtype=torch.float32) 
    
    # Use dimensions reported by the model logs to avoid mismatches
    state_dim = 23 # Matches PROPRIO_DIM in logs
    state = torch.randn((B, state_dim), dtype=torch.float32)
    state = state.unsqueeze(1) # Framework does this: [B, 1, state_dim]
    
    print(f"--- Starting Trace ---")
    with torch.no_grad():
        actions = action_model.predict_action(vl_embs, state)
    print(f"--- Trace Finished ---")
    print(f"Final actions shape: {actions.shape}")

if __name__ == "__main__":
    trace()
