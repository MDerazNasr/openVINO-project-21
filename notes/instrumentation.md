# UnifoLM-VLA Instrumentation Trace

## Overview
This document details the instrumentation added to the UnifoLM-VLA codebase to trace data flow, tensor shapes, and data types during inference. This work was completed during **Saturday Block 3**.

## Branch Information
- **Branch**: `trace/inference-shapes`
- **Purpose**: Lightweight diagnostic instrumentation to confirm architectural assumptions before OpenVINO conversion.

## Modified Files (Debug Prints)

### 1. `src/unifolm_vla/model/framework/unifolm_vla.py`
- Added trace for the Vision-Language Model (VLM) output.
- **Tensors**: `last_hidden` (hidden states from Qwen backbone).

### 2. `src/unifolm_vla/model/modules/action_model/DiT_ActionHeader.py`
- **Standardized Helper**: Introduced `_trace_tensor(name, x)` to log shape, dtype, and device.
- **Denoising Step Headers**: Added headers for the first and last steps (e.g., `[TRACE] denoising step 1/4`).
- **Traced Tensors**: 
    - `vl_embs`, `state` (Inputs)
    - `initial actions` (Noise)
    - `state_features`, `action_features`, `future_tokens` (Embeddings)
    - `sa_embs` (Joint sequence)
    - `model_output`, `pred`, `pred_velocity` (Predictions)
    - `updated actions` (Post-integration)

### 3. `src/unifolm_vla/model/modules/action_model/flow_matching_modules/cross_attention_dit.py`
- Added hooks for internal DiT forward pass monitoring.

## Trace Utility Script
- **File**: `scripts/trace_shapes.py`
- **Functionality**:
    - Loads the model configuration using `OmegaConf`.
    - Instantiates the `FlowmatchingActionHead` in isolation.
    - Mocks VLM embeddings and robot state tensors.
    - Executes a full `predict_action` cycle.
- **Usage**:
  ```bash
  export PYTHONPATH=$PYTHONPATH:./openvino-vla/unifolm-vla/src
  python3 scripts/trace_shapes.py
  ```

## Captured Inference Shapes (Typical Configuration)
| Tensor | Shape | Notes |
|---|---|---|
| `last_hidden` | `[1, 512, 2048]` | VLM Output (Batch, SeqLen, Dim) |
| `initial_actions` | `[1, 25, 23]` | Starting noise for 25-step horizon |
| `action_features` | `[1, 25, 1536]` | Encoded via `ActionEncoder` |
| `sa_embs` | `[1, 58, 1536]` | `[state(1) + queries(32) + actions(25)]` |
| `DiT_output` | `[1, 58, 1024]` | Output from Transformer blocks |
| `pred_velocity` | `[1, 25, 23]` | Predicted flow vector |
| `final_actions` | `[1, 25, 23]` | Integrated result |
