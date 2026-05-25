# Export Blocker Report

## Blocker 1 — Python denoising loop
Location: DiT_ActionHeader.py:251

Problem:
- `predict_action()` loops over `num_inference_timesteps`.
- If exported as one graph, tracing may unroll the DiT N times.
- Better approach: export a single DiT denoising step and keep the loop outside the graph initially.

Experiment:
- Try exporting full `predict_action()`.
- Try exporting single-step wrapper.

## Fix Strategy — Single-Step Wrapper

The intended fix is to create a wrapper that encapsulates exactly one iteration of the Euler integration step.

**Inputs**:
- `actions`: Current action trajectory [B, Horizon, ActionDim]
- `vl_embs`: VLM context embeddings [B, L, H]
- `state`: Robot proprioception [B, 1, StateDim]
- `timestep`: Scalar discrete timestep [B]

**Output**:
- `updated_actions` (Post-Euler integration) or `pred_velocity` (Raw flow prediction)

**Advantages**:
- Prevents the tracer from unrolling the loop $N$ times into a massive static graph.
- Allows for dynamic selection of `num_inference_timesteps` at runtime (outside the model graph).
- Simplifies kernel optimization for the core DiT block.

## Blocker 2 — torch.randn inside inference path
Location: `DiT_ActionHeader.py:239`

**Problem**:
- Initial noise for the action trajectory is generated inside `predict_action`.
- Stochastic tensor creation inside a static graph can cause non-determinism during optimization or require complex OpenVINO extensions.

**Planned Fix**:
- Modify the export wrapper to accept the initial `actions` noise as an external input tensor.
- The caller (CPU side) will generate the noise once per action chunk and pass it into the compiled OpenVINO graph.

## Blocker 3 — autocast in forward path
Location: `unifolm_vla.py:46, 57, 85, 98`

**Problem**:
- `torch.autocast` context managers are used to handle `bfloat16` and `float32` mixed precision at runtime.
- These context managers are runtime-specific and may be ignored or cause errors during static graph tracing.

**Planned Fix**:
- Remove `autocast` from the export-specific wrappers.
- Explicitly cast tensors to required dtypes (e.g., `.to(torch.float32)`) at the graph boundaries.
- Leverage OpenVINO's native precision configuration (FP16/INT8) during model compilation.

## Blocker 4 — BatchFeature boundary
Location: `DiT_ActionHeader.py:186`

**Problem**:
- The model uses HuggingFace `BatchFeature` containers (essentially dictionaries) to pass data.
- Static graph tracers (TorchScript/ONNX) expect standard Tensors, Lists, or Tuples. Custom dict-like containers create boundary errors.

**Planned Fix**:
- Design the OpenVINO export wrappers with explicit tensor-only arguments.
- Unpack `BatchFeature` objects into a dictionary of primitive tensors before the export boundary.

## Single-step DiT Export Attempt

**Goal**:
Export only one DiT denoising step instead of the full iterative loop to avoid graph unrolling.

**Inputs**:
- `vl_embs` [1, 512, 2048]
- `actions` [1, 8, 7]
- `state` [1, 1, 8]
- `timesteps_tensor` [1]

**Output**:
- `pred_velocity` [1, 8, 7]

**Result**: **SUCCESS** ✅

**Details**:
- The model was successfully converted using `ov.convert_model`.
- The IR files (`single_step_dit.xml` and `single_step_dit.bin`) were generated in `export_tests/`.
- **Note**: The conversion requires the correct robot platform argument (e.g., `libero`) to be passed during execution to ensure model internal constants match the input shapes.

## What counts as success here

## IR Runtime Validation

Result:
- **Successfully loaded** single-step DiT OpenVINO IR.
- **Successfully compiled** on CPU.
- **Successfully ran** dummy inference with traced LIBERO shapes.
- **Output shape**: `(1, 8, 7)` (matches expected action chunk).

Latency:
- **CPU single-step latency**: 117.110 ms

## IR Operator Inspection

Top operator types:
- **Constant**: 453
- **Convert**: 272
- **Add**: 191
- **MatMul**: 123
- **MVN**: 33
- **Multiply**: 20
- **ScaledDotProductAttention**: 16
- **Gelu**: 16

Initial observations:
- **AdaLayerNorm** appears to be represented as a decomposed sequence:
    - `MVN` (Mean Variance Normalization) + `Multiply` + `Add` (for static LayerNorm).
    - `MatMul` + `Swish` + `VariadicSplit` (for timestep embedding logic).
    - Final modulation: `Multiply` + `Add` using chunks from the linear projection.
- **Attention** appears to be **fused** into the high-level `ScaledDotProductAttention` operator (16 occurrences matches the 16 DiT layers).
- This supports the later profiling/fusion work because:
    - We have a clear baseline for single-step latency (117ms).
    - The `ScaledDotProductAttention` is already optimized, but the 33 `MVN` ops + surrounding math for adaptive normalization show clear potential for fusion into a single `AdaLayerNorm` kernel to reduce memory overhead.

