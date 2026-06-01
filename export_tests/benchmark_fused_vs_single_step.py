import torch
import numpy as np
import openvino as ov
import time
import sys
import os

# Paths to the two models
SINGLE_IR = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/artifacts/openvino_ir/single_step_dit.xml"
FUSED_IR = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/artifacts/openvino_ir/fused_loop_dit.xml"

def benchmark(name, compiled, inputs, runs=50):
    print(f"[INFO] Benchmarking {name}...")
    # Warmup
    for _ in range(5):
        _ = compiled(inputs)
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = compiled(inputs)
        times.append((time.perf_counter() - start) * 1000)
    
    return float(np.mean(times))

def main():
    core = ov.Core()
    
    # 1. Setup Single-Step Benchmark (Looping 4 times in Python)
    single_model = core.read_model(SINGLE_IR)
    single_compiled = core.compile_model(single_model, "CPU")
    
    # LIBERO dummy inputs
    vl_embs = np.random.randn(1, 512, 2048).astype(np.float32)
    actions = np.random.randn(1, 8, 7).astype(np.float32)
    state = np.random.randn(1, 1, 8).astype(np.float32)
    timesteps = np.array([0], dtype=np.int64)
    
    # Mapping for single-step
    single_inputs = {}
    for inp in single_compiled.inputs:
        name = inp.get_any_name()
        if "vl_embs" in name: single_inputs[inp] = vl_embs
        elif "actions" in name: single_inputs[inp] = actions
        elif "state" in name: single_inputs[inp] = state
        else: single_inputs[inp] = timesteps

    def run_single_loop():
        curr_actions = actions.copy()
        dt = 1.0 / 4
        for t in range(4):
            # Input mapping update not required for raw compute timing
            # but we assume the same inputs for fair math comparison
            ov_out = list(single_compiled(single_inputs).values())[0]
            curr_actions = curr_actions + dt * ov_out
    
    # 2. Setup Fused Benchmark (Single call to OpenVINO)
    fused_model = core.read_model(FUSED_IR)
    fused_compiled = core.compile_model(fused_model, "CPU")
    
    fused_inputs = {}
    for inp in fused_compiled.inputs:
        name = inp.get_any_name()
        if "vl_embs" in name: fused_inputs[inp] = vl_embs
        elif "initial_noise" in name: fused_inputs[inp] = actions
        elif "state" in name: fused_inputs[inp] = state

    # Execute Benchmarks
    print("\n[INFO] Starting final measurements...")
    
    # 1. Single Step Loop (Python Orchestrated)
    print("[INFO] Benchmarking Python-Orchestrated Loop...")
    # Warmup
    for _ in range(5): run_single_loop()
    
    start = time.perf_counter()
    for _ in range(50):
        run_single_loop()
    single_ms = ((time.perf_counter() - start) / 50) * 1000
    
    # 2. Fused Call (OpenVINO Internalized)
    fused_ms = benchmark("OpenVINO Fused Loop", fused_compiled, fused_inputs)

    print("\n--- Strategy Latency Comparison ---")
    print(f"Python-Orchestrated Loop: {single_ms:.2f} ms")
    print(f"OpenVINO Fused Loop:       {fused_ms:.2f} ms")
    print(f"Speedup from Fusion:      {(single_ms/fused_ms - 1)*100:.1f}%")

if __name__ == "__main__":
    main()
