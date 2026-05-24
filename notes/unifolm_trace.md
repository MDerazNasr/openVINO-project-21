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

### 2. Evaluation Script (LIBERO)
- **File**: `openvino-vla/unifolm-vla/experiments/LIBERO/eval_libero.py`
- **Purpose**: Orchestrates the evaluation loop, loading the environment (LIBERO) and feeding observations to the inference class.

### 3. Real-Time Model Server
- **File**: `openvino-vla/unifolm-vla/deployment/model_server/run_real_eval_server.py`
- **Purpose**: Provides a networked interface for inference, likely used for low-latency physical robot control.

---

## Execution Flow (Trace)

1. **Initialization**:
   - `Unifolm_VLA.from_pretrained(...)` loads the weight and configuration.
   - The model is moved to `DEVICE` (currently `CPU` on this machine).
   - Weights are optionally cast to `bfloat16`.

2. **Inference Loop (`step`)**:
   - **Pre-processing**: 
     - Images are handled via a history buffer (`deque` with `maxlen=horizon`).
     - Proprioceptive state is normalized using statistics gathered from the dataset.
   - **Model Forward Pass**: 
     - The `predict_action` method of `Unifolm_VLA` is called.
     - This involves the Qwen-based Vision-Language Model processing the vision/text tokens.
   - **Post-processing**:
     - Raw action tokens are converted back to continuous robotic control values (unnormalization).

---

## OpenVINO Optimization Candidates

- **Model Conversion Target**: The `predict_action` method in `Unifolm_VLA` (defined in `src/unifolm_vla/model/framework/unifolm_vla.py`).
- **Input Tensors**:
    - `image`: Vision tokens or raw pixel data.
    - `state`: Proprioceptive robot state.
    - `text`: Task instructions.
- **Output Tensors**:
    - `normalized_actions`: Predicted robotic actions.

---

## Artifacts & Logs
- `openvino-vla/unifolm-vla/scripts/eval_scripts/run_eval_libero.sh` (Shell entry point)
- `openvino-vla/unifolm-vla/experiments/logs/` (Contains execution results/metrics)
