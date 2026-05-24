# unifolm-vla Trace Notes

## Repo
- URL: https://github.com/MDerazNasr/openVINO-project-21.git
- Commit: a34b70dea34459dd7ef7bed7e7c3914aeefda193

## Key Files
- DiT action head: `openvino-vla/unifolm-vla/src/unifolm_vla/model/modules/action_model/DiT_ActionHeader.py`
- Main VLA model: `openvino-vla/unifolm-vla/src/unifolm_vla/model/framework/unifolm_vla.py`
- Inference script: `openvino-vla/unifolm-vla/experiments/LIBERO/unifolm_vla_inference.py`

## Confirmed Export Blockers
1. **Python denoising loop**: 
   - `openvino-vla/unifolm-vla/src/unifolm_vla/model/modules/action_model/DiT_ActionHeader.py`: Line 251 (`for t in range(num_steps):`)
   - This iterative loop in the Flow Matching head will be captured as a static loop if traced, or need specialized handling for OpenVINO.

2. **torch.randn in inference path**:
   - `openvino-vla/unifolm-vla/src/unifolm_vla/model/modules/action_model/DiT_ActionHeader.py`: Line 239 (Initial noise generation for action prediction)
   - Random number generation inside the model graph can be problematic for some export formats or require fixed seeds.

3. **torch.autocast in forward**:
   - `openvino-vla/unifolm-vla/src/unifolm_vla/model/framework/unifolm_vla.py`: Lines 46, 57, 85, 98.
   - Used for both `bfloat16` and `float32`. Autocast regions are often ignored or cause issues during static graph export; explicit casting is usually preferred for OpenVINO.

4. **BatchFeature boundary**:
   - `openvino-vla/unifolm-vla/src/unifolm_vla/model/modules/action_model/DiT_ActionHeader.py`: Lines 14, 186, 187.
   - The use of HuggingFace `BatchFeature` objects as internal data structures can break the tracer as it expects standard Tensors/Tuples/Dicts.

## Questions / Unknowns
- How does the `FlowmatchingActionHead` handle dynamic horizons during export?
- Can `bfloat16` autocast regions be automatically mapped to OpenVINO's native FP16/BF16 support without manual re-casting?
- Is `git-lfs` available in the target deployment environment for downloading the necessary weights for `lerobot` (if enabled)?
