# IR Graph Comparison - Single-Step vs Fused DiT

## Purpose

This note documents the comparison between:

```text
single_step_dit.xml
fused_loop_dit.xml
```

The mentor asked us to analyze how the fused-loop graph differs from the single-step graph, rather than only reporting latency. The goal is to understand what OpenVINO does structurally when the 4-step denoising loop is unrolled into one IR.

## Method

Script:

```text
export_tests/compare_ir_graphs.py
```

The script:

1. Loads both OpenVINO IRs with `ov.Core().read_model`.
2. Counts all operation types.
3. Compares XML and BIN sizes.
4. Highlights attention-related and AdaLayerNorm/MVN-related ops.
5. Writes JSON and Markdown outputs under `benchmark_outputs/`.

## Artifact Size Comparison

| Artifact | XML bytes | BIN bytes | Total ops |
|---|---:|---:|---:|
| Single-step DiT | 713,379 | 1,123,486,632 | 1,345 |
| Fused-loop DiT | 2,315,214 | 1,123,486,626 | 4,270 |

Ratios:

| Metric | Ratio |
|---|---:|
| XML size | 3.25x |
| BIN size | 1.000000x |
| Op count | 3.17x |

## Main Finding

The fused-loop graph substantially increases graph structure, but it does not increase weight storage.

Interpretation:

```text
OpenVINO unrolls the repeated compute into a larger graph,
but the repeated graph nodes reference shared/deduplicated weight buffers.
```

This confirms the key deployment result:

```text
We get fused-loop runtime benefits without a 4x weight-memory penalty.
```

## Focus Op Comparison

| Op | Single | Fused | Delta | Ratio |
|---|---:|---:|---:|---:|
| Add | 191 | 711 | 520 | 3.72x |
| Concat | 8 | 22 | 14 | 2.75x |
| Convert | 272 | 676 | 404 | 2.49x |
| MVN | 33 | 132 | 99 | 4.00x |
| MatMul | 123 | 438 | 315 | 3.56x |
| Multiply | 20 | 92 | 72 | 4.60x |
| Reshape | 64 | 216 | 152 | 3.38x |
| ScaledDotProductAttention | 16 | 64 | 48 | 4.00x |
| Transpose | 64 | 208 | 144 | 3.25x |
| VariadicSplit | 17 | 68 | 51 | 4.00x |

## What This Says About Attention

`ScaledDotProductAttention` scales exactly 4x:

```text
single-step: 16
fused-loop: 64
```

Interpretation:

- The fused graph contains four explicit DiT denoising steps.
- Attention is structurally repeated across the unrolled graph.
- The attention pattern remains recognizable to OpenVINO as `ScaledDotProductAttention`.

Next profiling question:

```text
Do these SDPA ops dominate GPU runtime, or are surrounding matmul/layout/normalization kernels also significant?
```

## What This Says About AdaLayerNorm

AdaLayerNorm-related patterns remain decomposed:

```text
MVN + Multiply + Add + VariadicSplit
```

Important counts:

| Op | Single | Fused | Ratio |
|---|---:|---:|---:|
| MVN | 33 | 132 | 4.00x |
| VariadicSplit | 17 | 68 | 4.00x |
| Multiply | 20 | 92 | 4.60x |
| Add | 191 | 711 | 3.72x |

Interpretation:

- AdaLayerNorm is still represented as decomposed primitive ops.
- The pattern scales across the unrolled denoising steps.
- This supports AdaLayerNorm/MVN as a plausible optimization target, but not yet a proven hotspot.

Decision:

```text
Do not start kernel work from graph counts alone.
Use VTune to confirm whether these decomposed ops are costly at runtime.
```

## What This Says About Layout / Precision Overhead

`Convert`, `Transpose`, and `Reshape` also grow substantially:

| Op | Single | Fused | Ratio |
|---|---:|---:|---:|
| Convert | 272 | 676 | 2.49x |
| Transpose | 64 | 208 | 3.25x |
| Reshape | 64 | 216 | 3.38x |

Interpretation:

- The fused graph may still carry meaningful layout/shape/precision movement.
- VTune/OpenVINO profiling should check whether these become real runtime costs.
- If layout/reorder kernels show up strongly in GPU profiling, optimization may need to target layout strategy rather than only AdaLayerNorm.

## Top Structural Deltas

| Op | Single | Fused | Delta | Ratio |
|---|---:|---:|---:|---:|
| Constant | 453 | 1330 | 877 | 2.94x |
| Add | 191 | 711 | 520 | 3.72x |
| Convert | 272 | 676 | 404 | 2.49x |
| MatMul | 123 | 438 | 315 | 3.56x |
| Reshape | 64 | 216 | 152 | 3.38x |
| Transpose | 64 | 208 | 144 | 3.25x |
| Unsqueeze | 37 | 156 | 119 | 4.22x |
| MVN | 33 | 132 | 99 | 4.00x |
| Multiply | 20 | 92 | 72 | 4.60x |
| VariadicSplit | 17 | 68 | 51 | 4.00x |
| Gelu | 16 | 64 | 48 | 4.00x |
| ScaledDotProductAttention | 16 | 64 | 48 | 4.00x |

## Presentation Takeaway

The fused-loop graph is not a black box. Structurally:

1. OpenVINO sees the four denoising steps as a larger static graph.
2. Compute ops scale roughly with the number of unrolled steps.
3. Weights are shared/deduplicated, so the `.bin` stays flat.
4. Attention remains represented as `ScaledDotProductAttention`.
5. AdaLayerNorm remains decomposed into MVN/elementwise patterns.

This gives a concrete reason to move into VTune profiling:

```text
The graph shows plausible optimization targets.
VTune should decide which target matters at runtime.
```
