# Numeric Parity & Strict Validation

## Goal
Establish a 100% deterministic baseline to prove mathematical equivalence between the PyTorch implementation and the OpenVINO Intermediate Representation (IR).

## Theoretical Rationale
In deep learning systems, numerical drift can occur during model conversion due to differences in operator implementations, precision handling (e.g., `bfloat16` vs. `float32`), and non-deterministic layers like Dropout or random noise generation. 

By enforcing a strict global seed and disabling non-deterministic behavior (`model.eval()`), we ensure that any divergence found is purely a result of the OpenVINO conversion engine or precision casting, rather than random weight initialization. This allows us to quantify the "lossiness" of the export.

## Mentor Alignment
- **Transcript**: Mentors requested validating the function-level accuracy against PyTorch results.
- **Specific Targets**: 
    - Mean Standard Error (MSE) < 0.1%
    - Mean Absolute Error (MAE) < 1e-3

## Execution Details
- **Script**: `export_tests/compare_single_step_parity_v2.py`
- **Method**: Injected `torch.manual_seed(42)` and `np.random.seed(42)` before model instantiation to guarantee identical initial weights in both the live PyTorch wrapper and the previously exported IR.

## Failure Prediction
1. **Precision Mismatch**: If PyTorch uses `bfloat16` and OpenVINO uses `float32`, a minor MAE divergence is expected.
2. **Layer Fusion Variance**: If OpenVINO fuses layers like `LayerNorm` and `Scale/Shift`, the floating-point accumulation order may change, slightly affecting precision.
3. **Weight Initialization Leakage**: If the model architecture contains internal random state (e.g. `torch.randn` for time embeddings) that isn't seeded correctly, parity will fail.

## Presentation Highlights
- Show the table comparing MSE/MAE against mentor targets.
- Highlight the deterministic reproducibility of the baseline.
