import numpy as np
import torch
import openvino as ov
import sys
import os
from omegaconf import OmegaConf

# Add project src to path to resolve imports
sys.path.append(os.path.abspath("openvino-vla/unifolm-vla/src"))

from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model
from single_step_dit_wrapper import SingleStepDiTWrapper

IR_PATH = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/artifacts/openvino_ir/single_step_dit.xml"

def main():
    print("[INFO] Loading PyTorch model...")
    config_path = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/openvino-vla/unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_train.yaml"
    config = OmegaConf.load(config_path)
    
    # Instantiate the action model (DiT Flowmatching head)
    action_model = get_action_model(config=config)
    wrapper = SingleStepDiTWrapper(action_model)
    wrapper.eval() # Ensure dropout is disabled

    print("[INFO] Creating SAME dummy inputs used for trace...")
    torch.manual_seed(0)
    batch_size = 1
    seq_len = 512
    vl_dim = 2048
    action_horizon = 8
    action_dim = 7
    state_dim = 8

    vl_embs = torch.randn(batch_size, seq_len, vl_dim, dtype=torch.float32)
    actions = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32)
    state = torch.randn(batch_size, 1, state_dim, dtype=torch.float32)
    timesteps = torch.zeros(batch_size, dtype=torch.long)

    # 3. PyTorch output
    print("[INFO] Running PyTorch inference...")
    with torch.no_grad():
        torch_out_1 = wrapper(vl_embs, actions, state, timesteps).detach().cpu().numpy()
        torch_out_2 = wrapper(vl_embs, actions, state, timesteps).detach().cpu().numpy()
    
    pytorch_internal_diff = np.abs(torch_out_1 - torch_out_2).max()
    print(f"[INFO] PyTorch internal max diff (determinism check): {pytorch_internal_diff}")
    
    torch_out = torch_out_1

    # 4. OpenVINO output
    print("[INFO] Loading OpenVINO IR...")
    core = ov.Core()
    model = core.read_model(IR_PATH)
    compiled = core.compile_model(model, "CPU")

    ov_inputs = {}
    print("[INFO] Mapping inputs to OpenVINO...")
    
    # Track which torch tensors we have already mapped to avoid multiple assignments
    mapped_inputs = set()

    for inp in compiled.inputs:
        try:
            name = inp.get_any_name()
        except:
            name = f"input_{inp.index}"
            
        p_shape = inp.get_partial_shape()
        
        # Priority 1: Match by name if names are present
        if "vl_embs" in name:
            ov_inputs[inp] = vl_embs.numpy()
            print(f"  Mapped vl_embs to {name}")
        elif "actions" in name:
            ov_inputs[inp] = actions.numpy()
            print(f"  Mapped actions to {name}")
        elif "state" in name:
            ov_inputs[inp] = state.numpy()
            print(f"  Mapped state to {name}")
        elif "timestep" in name or "183" in name: # 183 was the ID in previous logs
            ov_inputs[inp] = timesteps.numpy()
            print(f"  Mapped timesteps to {name}")
        else:
            # Priority 2: Fallback to shape compatibility if names are generic
            if p_shape.compatible(ov.PartialShape(list(vl_embs.shape))) and "vl_embs" not in mapped_inputs:
                ov_inputs[inp] = vl_embs.numpy()
                mapped_inputs.add("vl_embs")
                print(f"  Fallback-mapped vl_embs to {name}")
            elif p_shape.compatible(ov.PartialShape(list(actions.shape))) and "actions" not in mapped_inputs:
                ov_inputs[inp] = actions.numpy()
                mapped_inputs.add("actions")
                print(f"  Fallback-mapped actions to {name}")
            elif p_shape.compatible(ov.PartialShape(list(state.shape))) and "state" not in mapped_inputs:
                ov_inputs[inp] = state.numpy()
                mapped_inputs.add("state")
                print(f"  Fallback-mapped state to {name}")
            elif p_shape.rank.get_length() == 1 and p_shape[0].is_static and p_shape[0].get_length() == batch_size:
                ov_inputs[inp] = timesteps.numpy()
                print(f"  Fallback-mapped timesteps to {name}")
            else:
                 raise RuntimeError(f"Could not map input: {name}, shape={p_shape}")

    print("[INFO] Running OpenVINO inference...")
    ov_out_dict = compiled(ov_inputs)
    ov_out = list(ov_out_dict.values())[0]

    # 5. Compare
    diff = np.abs(torch_out - ov_out)

    print("\n--- Numerical Parity Results ---")
    print("PyTorch output shape: ", torch_out.shape)
    print("OpenVINO output shape:", ov_out.shape)
    print("Max abs diff:         ", diff.max())
    print("Mean abs diff:        ", diff.mean())
    print("Allclose atol=1e-4:   ", np.allclose(torch_out, ov_out, atol=1e-4, rtol=1e-4))
    print("Allclose atol=1e-3:   ", np.allclose(torch_out, ov_out, atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    main()
