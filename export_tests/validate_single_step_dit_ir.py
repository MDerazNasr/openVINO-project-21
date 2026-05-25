import time
import numpy as np
import openvino as ov


IR_PATH = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/artifacts/openvino-ir/single_step_dit.xml"
# Fix path if necessary - checking against previous 'mv' command
import os
if not os.path.exists(IR_PATH):
    IR_PATH = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/artifacts/openvino_ir/single_step_dit.xml"


def main():
    core = ov.Core()

    print("[INFO] Available devices:", core.available_devices)

    print("[INFO] Reading IR...")
    model = core.read_model(IR_PATH)

    print("[INFO] Inputs:")
    for i, inp in enumerate(model.inputs):
        try:
            name = inp.get_any_name()
        except:
            name = f"input_{i}"
        print(f"  {name} {inp.partial_shape} {inp.get_element_type()}")

    print("[INFO] Outputs:")
    for i, out in enumerate(model.outputs):
        try:
            name = out.get_any_name()
        except:
            name = f"output_{i}"
        print(f"  {name} {out.partial_shape} {out.get_element_type()}")

    print("[INFO] Compiling on CPU...")
    compiled = core.compile_model(model, "CPU")

    print("[INFO] Creating dummy inputs from model shapes...")
    inputs = {}
    
    # LIBERO defaults for dynamic shapes
    libero_shapes = {
        "vl_embs": (1, 512, 2048),
        "actions": (1, 8, 7),
        "state": (1, 1, 8),
        "timesteps_tensor": (1,),
        "input_3": (1,) # fallback for timesteps
    }

    for i, inp in enumerate(compiled.inputs):
        try:
            name = inp.get_any_name()
        except:
            name = f"input_{i}"
            
        p_shape = inp.get_partial_shape()
        
        # Determine fixed shape for input generation
        if p_shape.is_static:
            shape = list(inp.shape)
        else:
            # Handle dynamic shape by using LIBERO defaults
            # Qwen inputs are (B, L, H), Actions (B, Horizon, ActionDim), State (B, 1, StateDim)
            if "vl_embs" in name:
                shape = (1, 512, 2048)
            elif "actions" in name:
                shape = (1, 8, 7)
            elif "state" in name:
                shape = (1, 1, 8)
            else:
                shape = (1,) # Fallback for scalar/timestep
            
        dtype = np.float32
        if "i64" in str(inp.element_type) or "i32" in str(inp.element_type):
            dtype = np.int64

        if dtype == np.int64:
            inputs[inp] = np.zeros(shape, dtype=dtype)
        else:
            inputs[inp] = np.random.randn(*shape).astype(dtype)

        print(f"  Input {name}: shape={inputs[inp].shape}, dtype={inputs[inp].dtype}")

    print("[INFO] Running inference...")
    # Warmup
    _ = compiled(inputs)
    
    start = time.perf_counter()
    outputs = compiled(inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print("[INFO] Outputs:")
    for out, value in outputs.items():
        try:
            name = out.get_any_name()
        except:
            name = "output"
        print(f"  {name}: {value.shape} {value.dtype}")

    print(f"[RESULT] CPU single-step latency: {elapsed_ms:.3f} ms")
    print("[SUCCESS] IR runtime validation passed.")


if __name__ == "__main__":
    main()
