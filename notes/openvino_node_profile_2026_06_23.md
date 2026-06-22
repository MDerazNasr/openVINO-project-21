# OpenVINO Node Profiling - June 23, 2026

## Context

VTune is not currently available on the Intel runner `PATH`, so we added an interim OpenVINO per-node profiling path:

```text
export_tests/openvino_node_profile.py
```

Workflow run:

```text
https://github.com/MDerazNasr/openVINO-project-21/actions/runs/27986235751
```

Artifact:

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
| CPU | 305.94 ms | 1221.48 ms | 870.01 ms | 1.40x | 1.15 | ok |
| GPU | 16.01 ms | 64.73 ms | 54.95 ms | 1.18x | 18.20 | ok |
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
101.48 ms
```

Top node types:

| Node Type | Calls | Real Total | Real Mean |
|---|---:|---:|---:|
| FullyConnected | 4380 | 515.851 ms | 0.118 ms |
| SDPA | 640 | 25.481 ms | 0.040 ms |
| MVN | 1320 | 12.405 ms | 0.009 ms |
| Add | 7100 | 2.163 ms | 0.000 ms |
| Transpose | 640 | 2.070 ms | 0.003 ms |
| Convert | 160 | 1.863 ms | 0.012 ms |
| Crop | 680 | 1.716 ms | 0.003 ms |
| Concat | 210 | 1.060 ms | 0.005 ms |

Interpretation:

- `FullyConnected` / MatMul dominates the profile.
- SDPA is visible but much smaller than the matmul-heavy linear layers.
- MVN is measurable but not the dominant runtime component in this OpenVINO profile.
- Elementwise and layout-like nodes are present, but much smaller by reported real time.

## Python Loop Profile

Profiled wall-clock mean:

```text
110.51 ms
```

Top node types:

| Node Type | Calls | Real Total | Real Mean |
|---|---:|---:|---:|
| FullyConnected | 4920 | 628.500 ms | 0.128 ms |
| SDPA | 640 | 21.983 ms | 0.034 ms |
| MVN | 1320 | 12.141 ms | 0.009 ms |
| Convert | 280 | 5.739 ms | 0.020 ms |
| Transpose | 640 | 2.118 ms | 0.003 ms |
| Add | 7600 | 1.316 ms | 0.000 ms |
| Concat | 280 | 1.085 ms | 0.004 ms |
| Crop | 680 | 0.772 ms | 0.001 ms |

Interpretation:

- The Python-loop mode also reports `FullyConnected` as dominant.
- The Python-loop path has more `FullyConnected` calls than fused-loop in this profile.
- `Convert` is higher in the Python-loop profile than fused-loop, which may reflect repeated per-step execution or graph boundary effects.
- MVN remains visible but small relative to matmul-heavy layers.

## What This Means For The Kernel Plan

The proposal discussed AdaLayerNorm/MVN fusion as a possible first kernel contribution. This OpenVINO profile does not rule that out, but it weakens the case for choosing it blindly as the first target.

Current evidence:

```text
FullyConnected / MatMul dominates.
SDPA is second.
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
