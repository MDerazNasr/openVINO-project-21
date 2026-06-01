import torch
import torch.nn as nn
import sys
import os
from omegaconf import OmegaConf

# Add project src to path
sys.path.append(os.path.abspath("openvino-vla/unifolm-vla/src"))
from unifolm_vla.model.modules.action_model.DiT_ActionHeader_v2 import FlowmatchingActionHead_v2

class FullLoopDiTWrapper(nn.Module):
    """
    Wraps the DiT action head and unrolls the denoising loop into a static graph.
    """
    def __init__(self, action_model, num_steps=4):
        super().__init__()
        self.action_model = action_model
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps

    def forward(self, vl_embs, initial_noise, state):
        actions = initial_noise
        batch_size = vl_embs.shape[0]
        
        # We unroll the loop here so the tracer captures all N steps
        for t in range(self.num_steps):
            t_cont = t / float(self.num_steps)
            t_tensor = torch.full((batch_size,), t_cont, device=vl_embs.device, dtype=vl_embs.dtype)
            
            # Predict velocity for this step
            pred_velocity = self.action_model.forward(vl_embs, actions, state, t=t_tensor)
            
            # Euler integration
            actions = actions + self.dt * pred_velocity
            
        return actions
