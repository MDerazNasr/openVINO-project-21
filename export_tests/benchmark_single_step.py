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
from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model
from single_step_dit_wrapper import SingleStepDiTWrapper

IR_PATH = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/artifacts/openvino_ir/single_step_dit.xml"

def benchmark(name, fn, warmup=5, runs=50):
    print(f"\n[INFO] Benchmarking {name}...")
    # Warmup
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)

    results = {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
    }
    
    print(f"  Mean:   {results['mean_ms']:.2f} ms")
    print(f"  Median: {results['median_ms']:.2f} ms")
    print(f"  Min:    {results['min_ms']:.2f} ms")
    print(f"  Max:    {results['max_ms']:.2f} ms")
    print(f"  StdDev: {results['std_ms']:.2f} ms")
    
    return results

def main():
    print("[INFO] Preparing Benchmarks...")
    
    # 1. Load PyTorch Model
    config_path = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/openvino-vla/unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_train.yaml"
    config = OmegaConf.load(config_path)
    action_model = get_action_model(config=config)
    pt_wrapper = SingleStepDiTWrapper(action_model)
    pt_wrapper.eval()
    
    # 2. Load OpenVINO Model
    core = ov.Core()
    ov_model = core.read_model(IR_PATH)
    compiled = core.compile_model(ov_model, "CPU")
    
    # 3. Setup Inputs (LIBERO defaults)
    batch_size = 1
    seq_len = 512
    vl_dim = 2048
    action_horizon = 8
    action_dim = 7
    state_dim = 8
    num_steps = 4
    num_timestep_buckets = 1000
    
    # PyTorch inputs
    torch.manual_seed(0)
    pt_vl_embs = torch.randn(batch_size, seq_len, vl_dim, dtype=torch.float32)
    pt_actions = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32)
    pt_state = torch.randn(batch_size, 1, state_dim, dtype=torch.float32)
    pt_timesteps = torch.zeros(batch_size, dtype=torch.long)
    
    # OpenVINO inputs
    np.random.seed(0)
    ov_vl_embs = np.random.randn(batch_size, seq_len, vl_dim).astype(np.float32)
    ov_actions = np.random.randn(batch_size, action_horizon, action_dim).astype(np.float32)
    ov_state = np.random.randn(batch_size, 1, state_dim).astype(np.float32)
    ov_timesteps = np.zeros(batch_size, dtype=np.int64)

    ov_inputs = {}
    for inp in compiled.inputs:
        name = inp.get_any_name() if inp.get_names() else "unknown"
        if "vl_embs" in name: ov_inputs[inp] = ov_vl_embs
        elif "actions" in name: ov_inputs[inp] = ov_actions
        elif "state" in name: ov_inputs[inp] = ov_state
        else: ov_inputs[inp] = ov_timesteps # timesteps fallback
        
    # --- Benchmark Functions ---

    def run_pytorch_single_step():
        with torch.no_grad():
            _ = pt_wrapper(pt_vl_embs, pt_actions, pt_state, pt_timesteps)

    def run_openvino_single_step():
        _ = compiled(ov_inputs)

    def run_openvino_external_loop():
        # Mimic the orchestration loop overhead
        actions = ov_actions.copy()
        dt = 1.0 / num_steps
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_disc = np.array([int(t_cont * num_timestep_buckets)], dtype=np.int64)
            
            # Update dynamic inputs
            for inp in compiled.inputs:
                name = inp.get_any_name() if inp.get_names() else "unknown"
                if "actions" in name: ov_inputs[inp] = actions
                elif "timestep" in name or "183" in name or inp.get_partial_shape().rank.get_length() == 1:
                    ov_inputs[inp] = t_disc
                    
            pred_velocity = list(compiled(ov_inputs).values())[0]
            actions = actions + dt * pred_velocity
            
    # --- Execute Benchmarks ---
    # We use 50 runs for individual steps, 20 runs for the full loop to save time
    benchmark("PyTorch Single-Step", run_pytorch_single_step, warmup=5, runs=50)
    benchmark("OpenVINO Single-Step", run_openvino_single_step, warmup=5, runs=50)
    benchmark(f"OpenVINO External Loop ({num_steps} steps)", run_openvino_external_loop, warmup=3, runs=20)

if __name__ == "__main__":
    main()
