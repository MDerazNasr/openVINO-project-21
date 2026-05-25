import torch
import numpy as np
import openvino as ov
import time
import sys
import os
from omegaconf import OmegaConf

# Add project src to path to resolve imports
sys.path.append(os.path.abspath("openvino-vla/unifolm-vla/src"))

import unifolm_vla.rlds_dataloader.constants as vla_constants

IR_PATH = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/artifacts/openvino_ir/single_step_dit.xml"

def main():
    print("[INFO] Loading OpenVINO IR...")
    core = ov.Core()
    model = core.read_model(IR_PATH)
    compiled = core.compile_model(model, "CPU")

    # Load configuration to get denoising parameters
    config_path = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/openvino-vla/unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_train.yaml"
    config = OmegaConf.load(config_path)
    
    # 1. Initialize Constants (Using LIBERO defaults for this test)
    batch_size = 1
    num_steps = config.framework.action_model.num_inference_timesteps # Usually 4-8
    num_timestep_buckets = config.framework.action_model.num_timestep_buckets # Usually 1000
    dt = 1.0 / num_steps
    
    # LIBERO dimensions
    seq_len = 512
    vl_dim = 2048
    action_horizon = 8
    action_dim = 7
    state_dim = 8

    # 2. Create Initial Inputs
    # Fixed VLM context and robot state for the duration of this action chunk
    vl_embs = np.random.randn(batch_size, seq_len, vl_dim).astype(np.float32)
    state = np.random.randn(batch_size, 1, state_dim).astype(np.float32)
    
    # Initial noisy actions trajectory (the starting point for denoising)
    actions = np.random.randn(batch_size, action_horizon, action_dim).astype(np.float32)
    
    print(f"\n--- Starting External OpenVINO Denoising Loop ---")
    print(f"Total Steps: {num_steps}, Horizon: {action_horizon}, Action Dim: {action_dim}")
    
    start_total = time.perf_counter()

    # 3. Iterative Denoising Loop (Orchestration on CPU/Python)
    for t in range(num_steps):
        t_start = time.perf_counter()
        
        # Calculate discrete timestep for embedding
        t_cont = t / float(num_steps)
        t_discretized = np.array([int(t_cont * num_timestep_buckets)], dtype=np.int64)

        # Prepare inputs for the compiled DiT step
        # Note: Input mapping is based on our parity script findings
        ov_inputs = {}
        for inp in compiled.inputs:
            name = inp.get_any_name()
            shape = tuple(inp.get_partial_shape().to_shape() if inp.get_partial_shape().is_static else [])
            
            if "vl_embs" in name:
                ov_inputs[inp] = vl_embs
            elif "actions" in name:
                ov_inputs[inp] = actions
            elif "state" in name:
                ov_inputs[inp] = state
            elif "timestep" in name or "183" in name or len(shape) == 1:
                ov_inputs[inp] = t_discretized
        
        # Execute single OpenVINO DiT denoising step
        ov_results = compiled(ov_inputs)
        pred_velocity = list(ov_results.values())[0] # Returns [B, Horizon, ActionDim]
        
        # Update actions using Euler integration (Refinement)
        actions = actions + dt * pred_velocity
        
        t_elapsed = (time.perf_counter() - t_start) * 1000
        if t == 0 or t == num_steps - 1:
            print(f"[STEP {t+1}/{num_steps}] Latency: {t_elapsed:.2f} ms, actions_norm: {np.linalg.norm(actions):.4f}")

    elapsed_total_ms = (time.perf_counter() - start_total) * 1000
    
    print(f"\n--- Denoising Loop Finished ---")
    print(f"Final actions shape:  {actions.shape}")
    print(f"Total loop latency:   {elapsed_total_ms:.2f} ms")
    print(f"Average step latency: {elapsed_total_ms / num_steps:.2f} ms")
    print(f"[SUCCESS] OpenVINO external denoising loop architecture validated.")

if __name__ == "__main__":
    main()
