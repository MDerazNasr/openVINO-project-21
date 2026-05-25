import torch
import numpy as np
import sys
from omegaconf import OmegaConf

# Import constants to see what's being used
import unifolm_vla.rlds_dataloader.constants as vla_constants
from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model

def trace():
    # Load config
    config = OmegaConf.load("openvino-vla/unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_train.yaml")
    
    # Instantiate action model
    action_model = get_action_model(config=config)
    action_model.eval()
    
    # Dimensions from the global constants
    B = 1
    L = 512
    H = 2048 # config.framework.qwenvl.vl_hidden_dim
    
    vl_embs = torch.randn((B, L, H), dtype=torch.float32) 
    
    state_dim = vla_constants.PROPRIO_DIM
    state = torch.randn((B, state_dim), dtype=torch.float32)
    state = state.unsqueeze(1)
    
    print(f"\n--- Starting Trace ({vla_constants.ROBOT_PLATFORM}) ---")
    print(f"Constants: Horizon={vla_constants.NUM_ACTIONS_CHUNK}, ActionDim={vla_constants.ACTION_DIM}, ProprioDim={vla_constants.PROPRIO_DIM}")
    
    with torch.no_grad():
        actions = action_model.predict_action(vl_embs, state)
    
    print(f"--- Trace Finished ---")
    print(f"Final actions shape: {list(actions.shape)}")

if __name__ == "__main__":
    trace()
