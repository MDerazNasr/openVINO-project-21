# End-to-End VLM Backbone Export

## Goal
Expand the OpenVINO conversion scope beyond the DiT action head to include the complete Vision-Language Model (VLM) backbone (Qwen2.5-VL) and its Vision Encoder.

## Theoretical Rationale
In a robotic inference pipeline, the VLM is the "slow brain" that performs high-level scene understanding and instruction grounding. While the DiT runs iteratively, the VLM runs once per observation. However, because the VLM is several orders of magnitude larger (~2B+ parameters), it still contributes a massive portion of the total latency. 

Converting the VLM to OpenVINO IR allows the engine to optimize the transformer attention kernels and utilize specialized hardware (like the Intel NPU or GPU) for the vision-language feature extraction.

## Mentor Alignment
- **Transcript**: Mentors requested an "overview of the BLA model rather than owning the GIT component" and specifically mentioned converting other components like "LM for VIT."
- **Strategy**: By exporting the VLM, we provide the mentors with the complete "End-to-End" picture they requested.

## Execution Details
- **Module**: `openvino-vla/unifolm-vla/src/unifolm_vla/model/modules/vlm/QWen2_5.py`
- **Component**: `QWen_VL_Interface`
- **Target**: Export the forward pass that produces `last_hidden_states`.

## Failure Prediction
1. **Dynamic Shape Conflicts**: LLMs typically use dynamic sequence lengths. OpenVINO's `convert_model` might require explicit `DynamicAxes` hints to avoid crashing on the attention mask.
2. **Vision Grid Logic**: Qwen2.5-VL uses a complex `image_grid_thw` input to handle variable resolution. This multi-input structure is a common failure point for standard tracers.
3. **Memory OOM**: Exporting a 2B+ parameter model might exceed local RAM limits during the TorchScript tracing phase.

## Results Achieved
- **Pipeline Built**: Created `export_tests/convert_qwen_vlm.py` which isolates the `last_hidden_state` forward pass of the Qwen2.5-VL backbone.
- **Export Validation**: Successfully ran the OpenVINO conversion pipeline using a structural mock (due to local RAM/weight access constraints).
- **IR Status**: Generated a template IR for the VLM feature extraction.

## June 22 Update — Mock Export Guardrail

The VLM conversion script was updated so mock fallback is no longer the default behavior.

Reason:
- The earlier mock/template export was useful for validating the conversion pipeline shape.
- It is not valid for VLM latency, full VLA latency, or trained-model claims.
- The mock `qwen_vlm_backbone.bin` artifact was tiny, not a real Qwen2.5-VL weight file.

Current behavior:
- `python export_tests/convert_qwen_vlm.py` now requires the real Qwen2.5-VL interface to load.
- If real model loading fails, the script raises an error.
- `--allow-mock` must be passed explicitly for structural tests.

This prevents us from accidentally reporting a fake end-to-end VLA benchmark.

## Findings
The Qwen2.5-VL backbone is significantly more complex to export than the DiT due to:
1. **Multimodal Interface**: Requires four distinct input tensors (`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`).
2. **Weight Memory**: The model weights exceed 10GB, requiring high-memory Intel DevCloud instances or large local workstations for the final export.

## Conclusion
The E2E export infrastructure is complete. The next step is to execute the script on a high-memory machine with access to the official Qwen2.5-VL weights.
