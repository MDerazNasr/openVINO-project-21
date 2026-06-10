# Export Blocker Report — unifolm-vla OpenVINO

## Repository
- URL: https://github.com/MDerazNasr/openVINO-project-21.git
- Commit: d2bd049338a0ae09b0879a110893d7ec4c633fd0

## Summary
I traced the inference path and confirmed that the VLA action generation path is:

Qwen2.5-VL embeddings → DiT action head → iterative denoising loop → action chunk.

The main export strategy is to avoid exporting the full Python denoising loop and instead export a single DiT denoising step with explicit tensor inputs.

## Blocker 1 — Python Denoising Loop
Location:
- `DiT_ActionHeader.py:251`

Problem:
- Full `predict_action()` loops over denoising timesteps in Python.
- Exporting the full loop risks unrolling the DiT N times into one large static graph.

Experiment:
- Created `SingleStepDiTWrapper`.
- Exported one DiT denoising step.

Result:
- Success: generated OpenVINO IR for single-step DiT.

Implication:
- Core DiT compute graph is exportable.
- Main blocker is orchestration/control flow, not the transformer compute graph.

Next:
- Keep denoising loop outside the exported graph initially.
- Reuse compiled single-step DiT graph across timesteps.

## Blocker 2 — torch.randn In Inference Path
Location:
- `DiT_ActionHeader.py:239`

Problem:
- `predict_action()` creates initial action noise internally.
- Random tensor generation should not be inside exported graph boundary.

Fix Direction:
- Accept initial action noise as explicit input to wrapper.

Status:
- Addressed in single-step wrapper design.

## Blocker 3 — autocast In Forward Path
Location:
- `unifolm_vla.py:46,57,85,98`

Problem:
- Precision context managers are embedded inside model logic.
- OpenVINO export should use explicit tensor dtypes and runtime/compile precision settings.

Fix Direction:
- Export wrapper avoids autocast.
- Later use OpenVINO compile configs for BF16/FP32 behavior.

Status:
- Partially addressed for single-step wrapper.

## Blocker 4 — BatchFeature Boundary
Location:
- `DiT_ActionHeader.py:186`

Problem:
- HF `BatchFeature`/dict containers are not clean graph inputs.

Fix Direction:
- Use tensor-in/tensor-out wrapper boundaries.

Status:
- Addressed for single-step DiT wrapper.

## IR Validation
- IR generated: Yes (`single_step_dit.xml`, `single_step_dit.bin`)
- IR loaded: Yes
- CPU compile: Yes
- Dummy inference: Yes
- Output shape: `(1, 8, 7)`
- CPU latency (Single-Step): 105.97 ms
- Numerical Parity: **PASS** (MSE: 0.00059%, below 0.1% target)

## Operator Inspection
Top operators:
- Constant: 453
- Convert: 272
- Add: 191
- MatMul: 123
- MVN: 33
- ScaledDotProductAttention: 16

AdaLayerNorm representation:
- Decomposed into `MVN` (33 occurrences), `Multiply`, `Add`, and `VariadicSplit` logic.

Attention representation:
- Fused into `ScaledDotProductAttention` (16 occurrences).

## Current Conclusion
The single-step DiT export path is viable and numerically accurate. However, benchmarking reveals that a **Fused Loop** (unrolled 4-step graph) provides a significant performance advantage over a Python-orchestrated loop (259ms vs 495ms, ~91% speedup). Crucially, the Fused Loop does **not** duplicate weights, keeping the memory footprint at ~1.0GB (same as single-step). Future work will focus on finalizing the fused loop for production while maintaining the single-step IR for research flexibility.
