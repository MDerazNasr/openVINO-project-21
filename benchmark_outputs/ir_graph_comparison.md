# OpenVINO IR Graph Comparison

## Artifact Sizes

| Artifact | XML bytes | BIN bytes | Total ops |
|---|---:|---:|---:|
| Single-step DiT | 713379 | 1123486632 | 1345 |
| Fused-loop DiT | 2315214 | 1123486626 | 4270 |

- XML size ratio: `3.25x`
- BIN size ratio: `1.000000x`
- Op count ratio: `3.17x`

## Focus Ops

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
| Softmax | 0 | 0 | 0 | n/a |
| Transpose | 64 | 208 | 144 | 3.25x |
| VariadicSplit | 17 | 68 | 51 | 4.00x |

## Top Op Deltas

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
| Concat | 8 | 22 | 14 | 2.75x |
| Gather | 4 | 18 | 14 | 4.50x |
| Slice | 3 | 12 | 9 | 4.00x |
| Swish | 3 | 12 | 9 | 4.00x |
| Broadcast | 2 | 9 | 7 | 4.50x |
| Cos | 2 | 8 | 6 | 4.00x |
| Sin | 2 | 8 | 6 | 4.00x |
| ShapeOf | 2 | 6 | 4 | 3.00x |
| Exp | 1 | 4 | 3 | 4.00x |
| Range | 1 | 4 | 3 | 4.00x |
| Relu | 2 | 5 | 3 | 2.50x |
| Parameter | 4 | 3 | -1 | 0.75x |
| Squeeze | 1 | 0 | -1 | 0.00x |

## Interpretation Prompts

- If `.bin` size stays flat while XML/op count grows, OpenVINO is sharing weights across unrolled steps.
- If attention ops scale with the loop count, the fused graph is structurally unrolled rather than a dynamic loop.
- If MVN/Add/Multiply scale strongly, AdaLayerNorm remains decomposed and may be a fusion target if profiling confirms runtime impact.
- If Convert/Transpose/Reshape grow substantially, layout and precision transformation overhead should be inspected.