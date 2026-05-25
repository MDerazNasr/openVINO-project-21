# UnifoLM-VLA Inference Architecture & Trace

## Primary Inference Entry Points

### 1. Core Inference Class
- **File**: `openvino-vla/unifolm-vla/experiments/LIBERO/unifolm_vla_inference.py`
- **Class**: `Unifolm_VLA_Inference`
- **Purpose**: Wraps the raw model into a step-based interface suitable for RL environments or real robots.
- **Key Method**: `step(obs_inputs)`
    - Performs normalization of proprioceptive state.
    - Executes `self.vla.predict_action(qwen_inputs)`.
    - Unnormalizes output actions back to physical units.

### 2. Model Implementation Framework
- **File**: `openvino-vla/unifolm-vla/src/unifolm_vla/model/framework/unifolm_vla.py`
- **Class**: `Unifolm_VLA`
- **Architecture**:
    - **VLM Backbone**: Qwen-VL (via `get_vlm_model`).
    - **Action Head**: `FlowmatchingActionHead` (a Flow Matching DiT - Diffusion Transformer).
- **Inference Logic (`predict_action`)**:
    1. **VLM Forward**: Passes `input_ids`, `pixel_values`, etc., through Qwen-VL.
    2. **Hidden State Extraction**: Extracts `last_hidden_states[-1]` from the VLM.
    3. **Action Prediction**: Passes the VLM's last hidden state and the robot `state` (proprioception) to `self.action_model.predict_action`.
    4. **Denoising**: The denoising/inference loop is internal to the `FlowmatchingActionHead`.

### 3. Supporting Scripts
- **Evaluation Script (LIBERO)**: `openvino-vla/unifolm-vla/experiments/LIBERO/eval_libero.py`
- **Model Server**: `openvino-vla/unifolm-vla/deployment/model_server/run_real_eval_server.py`

---

## Execution Flow (Trace)

1. **Initialization**:
   - `Unifolm_VLA.from_pretrained(...)` initializes the Qwen-VL backbone and the Flow Matching head.
   - The model is moved to `DEVICE` (currently `CPU`) and set to `eval()` mode.

2. **Inference Loop (`step`)**:
   - **Pre-processing**: 
     - Proprioceptive state is normalized.
     - VLM-specific inputs (`input_ids`, `pixel_values`, `image_grid_thw`) are prepared.
   - **Model Forward Pass**: 
     - **VLM Phase**: Qwen-VL processes multimodal inputs to produce a rich hidden representation.
     - **Action Phase**: `FlowmatchingActionHead` uses the VLM representation as a condition to predict actions via a flow-matching (diffusion-like) process.
   - **Post-processing**:
     - Actions are unnormalized and returned for robot execution.

---

## OpenVINO Optimization Candidates

- **VLM Pipeline**: Qwen-VL is the most compute-intensive part. Converting the transformer backbone to OpenVINO (using `optimum-intel` or `ovc`) is a priority.
- **Flow Matching Head**: `FlowmatchingActionHead` contains the denoising/inference loop. This might require custom export logic if it involves iterative loops.
- **Data Types**: The original code uses `bfloat16` for the VLM and `float32` for the action head.

## Inference Call Chain
1. **LIBERO script starts at**: `openvino-vla/unifolm-vla/experiments/LIBERO/eval_libero.py` (which orchestrates the rollout loop).
2. **Model loaded by**: `Unifolm_VLA.from_pretrained` (called in the `__init__` of `Unifolm_VLA_Inference` within `unifolm_vla_inference.py`).
3. **Input batch prepared by**: `Unifolm_VLA_Inference.step` (performs proprioception normalization and tensor conversion).
4. **Main VLA forward/action call**: `Unifolm_VLA.predict_action` in `openvino-vla/unifolm-vla/src/unifolm_vla/model/framework/unifolm_vla.py`.
5. **DiT action head call**: `FlowmatchingActionHead.predict_action` in `openvino-vla/unifolm-vla/src/unifolm_vla/model/modules/action_model/DiT_ActionHeader.py`.
6. **Denoising loop**: `for t in range(num_steps):` at line 251 of `DiT_ActionHeader.py`.
7. **Final action output**: The `step` method in `unifolm_vla_inference.py` unnormalizes the model's output and returns the physical robot actions.

## Current Architecture Understanding

Input observation / image / instruction
    ↓
LIBERO inference script
    ↓
UnifolmVLA model wrapper
    ↓
VLM backbone / feature extraction
    ↓
Action head preparation
    ↓
DiT denoising action model
    ↓
Action chunk output

## Core Functions

| File | Function | Role |
|---|---|---|
| unifolm_vla.py | forward | training/eval forward path |
| unifolm_vla.py | predict_action | inference action prediction entry |
| DiT_ActionHeader.py | prepare_input | converts batch dict to BatchFeature |
| DiT_ActionHeader.py | forward | training flow-matching loss path |
| DiT_ActionHeader.py | predict_action | inference denoising loop |
| cross_attention_dit.py | forward methods | DiT blocks / attention execution |
| QWen2_5.py | forward | VLM feature extraction |

## CrossAttentionDiT Classes
- **Class 1**: `TimestepEncoder` (Line 16) - Encodes discrete timesteps into continuous embeddings.
- **Class 2**: `AdaLayerNorm` (Line 29) - Adaptive Layer Normalization conditioned on timestep embeddings.
- **Class 3**: `BasicTransformerBlock` (Line 55) - The fundamental compute unit containing self/cross-attention and feed-forward layers.
- **Class 4**: `DiT` (Line 175) - The main Diffusion Transformer model that chains multiple transformer blocks.
- **Class 5**: `SelfAttentionTransformer` (Line 290) - A variant of the transformer focused on self-attention.

---

## Artifacts & Logs
- `openvino-vla/unifolm-vla/scripts/eval_scripts/run_eval_libero.sh`
- `openvino-vla/unifolm-vla/experiments/logs/`
