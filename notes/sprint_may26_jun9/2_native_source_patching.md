# Native Source Patching (GSoC Milestone)

## Goal
Directly modify the `unifolm-vla` source code to eliminate OpenVINO export blockers (`BatchFeature`, `torch.randn`, and `autocast`) at the architectural level, rather than relying on external wrappers.

## Theoretical Rationale
While external wrappers can bypass blockers for a single export, they create a fragile "bridge" that must be maintained separately from the core model logic. 

**Native Patching** ensures that the model architecture itself becomes **Export-First**. By removing HuggingFace-specific containers (`BatchFeature`) and externalizing stochastic noise generation (`torch.randn`), we create a "Clean Graph Boundary" where the model's interface is composed strictly of standard Tensors. This is essential for robust deployment on Intel iGPU/NPU, where any non-standard container can crash the compiler's fusion passes.

## Mentor Alignment
- **Transcript**: Mentors specifically suggested "patching the code" to "push boundaries" regarding `BatchFeature`.
- **Strategy**: They advised that the compiler can optimize better if the boundaries are clean and the graph is large.

## Execution Details
- **Module**: `openvino-vla/unifolm-vla/src/unifolm_vla/model/modules/action_model/DiT_ActionHeader.py`
- **Fix 1**: Removed `BatchFeature` import and replaced `prepare_input` with a standard dictionary return.
- **Fix 2**: Modified `predict_action` and `forward` to accept an optional `t` tensor and pre-generated noise, removing internal stochastic dependency.

## Failure Prediction
1. **API Breakage**: External scripts that rely on the `BatchFeature` return type from `prepare_input` will fail.
2. **Type Checking**: If static type checkers (Mypy) are used, the change from `BatchFeature` to `dict` might cause linting errors.
3. **Training Path Impact**: If the `randn` fix isn't carefully gated, it could interfere with the random noise sampling required for training.

## Presentation Highlights
- Show the code diff of the `BatchFeature` removal.
- Explain how "Pushing Boundaries" into the source code makes the VLA model natively compatible with OpenVINO GenAI.
