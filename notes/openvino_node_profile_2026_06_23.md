# OpenVINO Node Profiling - June 23, 2026

## Context

VTune is not currently available on the Intel runner `PATH`, so we added an interim OpenVINO per-node profiling path:

```text
export_tests/openvino_node_profile.py
```

Latest category-profile workflow run:

```text
https://github.com/MDerazNasr/openVINO-project-21/actions/runs/27986729224
```

Latest category-profile artifact:

```text
https://github.com/MDerazNasr/openVINO-project-21/actions/runs/27986729224/artifacts/7806038666
```

Earlier first-pass workflow run:

```text
https://github.com/MDerazNasr/openVINO-project-21/actions/runs/27986235751
```

Earlier first-pass artifact:

```text
https://github.com/MDerazNasr/openVINO-project-21/actions/runs/27986235751/artifacts/7805855968
```

The workflow succeeded, including:

- multi-device benchmark,
- deep hardware benchmark,
- OpenVINO node profiling,
- VTune availability check.

VTune remains unavailable:

```text
VTune CLI was not found on PATH.
```

## Benchmark Summary From Same Run

| Device | Single Step Mean | Python Loop Mean | Fused Loop Mean | Fused Speedup | Fused Chunks/s | Status |
|---|---:|---:|---:|---:|---:|---|
| CPU | 306.02 ms | 1253.17 ms | 869.47 ms | 1.44x | 1.15 | ok |
| GPU | 15.93 ms | 63.88 ms | 54.48 ms | 1.17x | 18.36 | ok |
| NPU | n/a | n/a | n/a | n/a | n/a | skipped |

Full VLA latency is still blocked:

```text
No Qwen2.5-VL .bin artifact is present; VLM and full VLA latency are blocked.
```

## Profiling Method

The script compiles the DiT IR with:

```text
PERF_COUNT = YES
```

and profiles two GPU modes:

1. `fused_loop_4_step`
2. `python_loop_4_step`

The profiling run used:

```text
iterations = 10
device = GPU
```

Important caveat:

OpenVINO profiling mode changes runtime behavior and adds overhead. The profiled wall-clock means are therefore not the benchmark latency numbers. Use this output for hotspot ranking, not final latency reporting.

## Fused Loop Profile

Profiled wall-clock mean:

```text
101.41 ms
```

Top categories:

| Category | Calls | Real Total | Share | Real Mean |
|---|---:|---:|---:|---:|
| mlp_fully_connected | 1280 | 293.688 ms | 52.15% | 0.229 ms |
| self_attention_projection | 2080 | 146.464 ms | 26.01% | 0.070 ms |
| other_fully_connected | 720 | 65.484 ms | 11.63% | 0.091 ms |
| attention_sdpa | 640 | 24.913 ms | 4.42% | 0.039 ms |
| normalization_mvn | 1320 | 12.466 ms | 2.21% | 0.009 ms |
| layout_shape_data_movement | 6480 | 7.007 ms | 1.24% | 0.001 ms |
| action_encoder_fully_connected | 120 | 6.386 ms | 1.13% | 0.053 ms |
| timestep_fully_connected | 80 | 2.472 ms | 0.44% | 0.031 ms |
| elementwise | 8190 | 2.421 ms | 0.43% | 0.000 ms |
| action_decoder_fully_connected | 80 | 1.353 ms | 0.24% | 0.017 ms |
| state_encoder_fully_connected | 20 | 0.423 ms | 0.08% | 0.021 ms |
```

Top node types:

| Node Type | Calls | Real Total | Real Mean |
|---|---:|---:|---:|
| FullyConnected | 4380 | 516.270 ms | 0.118 ms |
| SDPA | 640 | 24.913 ms | 0.039 ms |
| MVN | 1320 | 12.466 ms | 0.009 ms |
| Add | 7100 | 2.193 ms | 0.000 ms |
| Transpose | 640 | 2.051 ms | 0.003 ms |
| Convert | 160 | 1.887 ms | 0.012 ms |
| Crop | 680 | 1.678 ms | 0.002 ms |
| Concat | 210 | 1.059 ms | 0.005 ms |

Interpretation:

- `FullyConnected` / MatMul dominates the profile.
- The largest category is MLP fully-connected work, followed by self-attention projections.
- SDPA is visible but much smaller than the matmul-heavy linear layers.
- MVN is measurable at about 2.21% of profiled real time, but it is not the dominant runtime component in this OpenVINO profile.
- Elementwise and layout-like nodes are present, but much smaller by reported real time.

## Python Loop Profile

Profiled wall-clock mean:

```text
110.29 ms
```

Top categories:

| Category | Calls | Real Total | Share | Real Mean |
|---|---:|---:|---:|---:|
| mlp_fully_connected | 1280 | 295.533 ms | 43.67% | 0.231 ms |
| self_attention_projection | 2560 | 245.462 ms | 36.27% | 0.096 ms |
| other_fully_connected | 720 | 74.346 ms | 10.99% | 0.103 ms |
| attention_sdpa | 640 | 22.943 ms | 3.39% | 0.036 ms |
| normalization_mvn | 1320 | 12.149 ms | 1.80% | 0.009 ms |
| layout_shape_data_movement | 7000 | 10.449 ms | 1.54% | 0.001 ms |
| action_encoder_fully_connected | 120 | 7.359 ms | 1.09% | 0.061 ms |
| timestep_fully_connected | 80 | 2.767 ms | 0.41% | 0.035 ms |
| elementwise | 8600 | 2.269 ms | 0.34% | 0.000 ms |
| state_encoder_fully_connected | 80 | 1.875 ms | 0.28% | 0.023 ms |
| action_decoder_fully_connected | 80 | 1.413 ms | 0.21% | 0.018 ms |
```

Top node types:

| Node Type | Calls | Real Total | Real Mean |
|---|---:|---:|---:|
| FullyConnected | 4920 | 628.755 ms | 0.128 ms |
| SDPA | 640 | 22.943 ms | 0.036 ms |
| MVN | 1320 | 12.149 ms | 0.009 ms |
| Convert | 280 | 5.761 ms | 0.021 ms |
| Transpose | 640 | 2.173 ms | 0.003 ms |
| Add | 7600 | 1.343 ms | 0.000 ms |
| Concat | 280 | 1.114 ms | 0.004 ms |
| Crop | 680 | 0.771 ms | 0.001 ms |

Interpretation:

- The Python-loop mode also reports `FullyConnected` as dominant.
- The largest Python-loop categories are MLP fully-connected work and self-attention projections.
- The Python-loop path has more `FullyConnected` calls than fused-loop in this profile.
- `Convert` is higher in the Python-loop profile than fused-loop, which may reflect repeated per-step execution or graph boundary effects.
- MVN remains visible but small relative to matmul-heavy layers.

## What This Means For The Kernel Plan

The proposal discussed AdaLayerNorm/MVN fusion as a possible first kernel contribution. This OpenVINO profile does not rule that out, but it weakens the case for choosing it blindly as the first target.

Current evidence:

```text
MLP FullyConnected / MatMul is the largest bucket.
Self-attention projections are the second-largest bucket.
SDPA is visible but smaller.
MVN is measurable but much smaller.
```

Practical conclusion:

1. Do not claim AdaLayerNorm/MVN is the primary bottleneck yet.
2. Use VTune when available to validate lower-level GPU utilization, memory bandwidth, and stalls.
3. In the meantime, inspect whether the dominant `FullyConnected` nodes map to attention projections, MLP layers, or both.
4. Compare this with FLOPs/roofline analysis to see whether the model is compute-bound in practice.

## Presentation-Safe Summary

Safe wording:

```text
Because VTune was unavailable on the runner, I added OpenVINO runtime profiling as an interim hotspot view. The first node-level profile suggests the DiT action head is dominated by matmul/FullyConnected operations, with SDPA and MVN visible but much smaller. This means the first kernel contribution should still be decided after VTune or deeper profiling, rather than assuming AdaLayerNorm/MVN is the main bottleneck.
```

Do not say:

```text
VTune proves FullyConnected is the bottleneck.
```

We did not run VTune yet.

## Next Steps

- [ ] Locate or install VTune on the Intel runner.
- [ ] If VTune remains blocked, extend OpenVINO profiling analysis:
  - group top nodes by transformer block,
  - separate attention projections from MLP layers,
  - compare fused-loop vs Python-loop graph-boundary overhead.
- [ ] Update presentation slide:
  - "Interim OpenVINO profile: MatMul/FullyConnected dominates."
- [ ] Ask mentor whether to prioritize VTune setup, MatMul/attention investigation, or continue with the proposed AdaLayerNorm/MVN contribution path.
