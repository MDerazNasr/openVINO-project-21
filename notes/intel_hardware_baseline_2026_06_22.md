# Intel Hardware Baseline - June 22, 2026

## Summary

This run establishes the first successful Intel target-hardware baseline for the DiT action-head portion of the UnifoLM-VLA OpenVINO project.

The benchmark was executed through a GitHub Actions self-hosted runner on an Intel Core Ultra system. The runner regenerated both OpenVINO IR weight files locally because the GitHub checkout does not include `.bin` artifacts.

## Hardware and Runtime

| Component | Value |
|---|---|
| Runner | `PDX88-K0660` |
| CPU | Intel(R) Core(TM) Ultra 7 258V |
| GPU | Intel(R) Arc(TM) 140V GPU (16GB) iGPU |
| NPU | Intel(R) AI Boost |
| OpenVINO | `2026.2.1-21919-ede283a88e3-releases/2026/2` |
| Python | `3.12.10` |
| Available OpenVINO devices | `['CPU', 'GPU', 'NPU']` |

Device capabilities reported by OpenVINO:

| Device | Capabilities |
|---|---|
| CPU | `BF16`, `FP32`, `FP16`, `INT8`, `BIN`, `EXPORT_IMPORT` |
| GPU | `FP32`, `BIN`, `FP16`, `INT8`, `GPU_HW_MATMUL`, `GPU_USM_MEMORY`, `EXPORT_IMPORT` |
| NPU | `FP16`, `INT8`, `EXPORT_IMPORT` |

## Generated IR Artifacts

| Artifact | Size |
|---|---:|
| `single_step_dit.bin` | 1,123,599,304 bytes |
| `single_step_dit.xml` | 737,156 bytes |
| `fused_loop_dit.bin` | 1,123,599,298 bytes |
| `fused_loop_dit.xml` | 2,392,711 bytes |

This confirms the previous weight-sharing finding on Intel hardware as well: the fused 4-step graph increases XML graph size, but the `.bin` weight file remains effectively flat versus the single-step IR.

## Benchmark Inputs

The run used the upstream/default G1 configuration constants:

| Constant | Value |
|---|---:|
| `NUM_ACTIONS_CHUNK` | 25 |
| `ACTION_DIM` | 23 |
| `PROPRIO_DIM` | 23 |

Input ports observed in OpenVINO:

Single-step DiT:

| Index | Name | Shape | Type |
|---:|---|---|---|
| 0 | `vl_embs` | `[?,?,?]` | `float32` |
| 1 | `actions` | `[?,?,?]` | `float32` |
| 2 | `state` | `[?,?,?]` | `float32` |
| 3 | `183` | `[?]` | `int64_t` |

Fused-loop DiT:

| Index | Name | Shape | Type |
|---:|---|---|---|
| 0 | `vl_embs` | `[?,?,?]` | `float32` |
| 1 | `initial_noise` | `[?,?,?]` | `float32` |
| 2 | `state` | `[?,?,?]` | `float32` |

## Quick Latency Results

| Device | Python-Orchestrated Single-Step Loop | Fused 4-Step IR | Fused Speedup |
|---|---:|---:|---:|
| CPU | 1215.52 ms | 874.07 ms | 1.39x |
| GPU | 69.02 ms | 57.87 ms | 1.19x |
| NPU | skipped | skipped | n/a |

## Deep Benchmark Results

The follow-up deep benchmark collected compile latency, first-call/warmup behavior, steady-state distributions, and throughput. It used 12 measured runs for CPU and 100 measured runs for GPU.

### Compile Time

| Device | Single-Step Compile | Fused-Loop Compile |
|---|---:|---:|
| CPU | 309.13 ms | 1083.28 ms |
| GPU | 934.65 ms | 2268.10 ms |

### Steady-State Latency Distribution

| Device | Strategy | Mean | Median | Min | Max | Std | P90 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | Single DiT step | 307.54 ms | 306.77 ms | 302.44 ms | 315.17 ms | 3.66 ms | 310.99 ms | 312.91 ms | 314.72 ms |
| CPU | Python 4-step loop | 1223.56 ms | 1223.50 ms | 1214.46 ms | 1230.23 ms | 4.28 ms | 1228.76 ms | 1229.49 ms | 1230.08 ms |
| CPU | Fused 4-step IR | 873.17 ms | 872.13 ms | 868.02 ms | 882.66 ms | 3.59 ms | 877.41 ms | 879.94 ms | 882.11 ms |
| GPU | Single DiT step | 16.20 ms | 16.12 ms | 15.83 ms | 17.34 ms | 0.31 ms | 16.66 ms | 16.85 ms | 17.16 ms |
| GPU | Python 4-step loop | 65.31 ms | 65.03 ms | 63.66 ms | 68.60 ms | 1.20 ms | 67.01 ms | 67.65 ms | 68.27 ms |
| GPU | Fused 4-step IR | 54.39 ms | 54.22 ms | 53.75 ms | 56.10 ms | 0.51 ms | 55.35 ms | 55.58 ms | 55.93 ms |

### Throughput

| Device | Python-Loop Chunks/s | Fused-Loop Chunks/s | Fused Speedup |
|---|---:|---:|---:|
| CPU | 0.82 | 1.15 | 1.40x |
| GPU | 15.31 | 18.39 | 1.20x |
| NPU | n/a | n/a | skipped |

### Warmup / First-Call Behavior

| Device | Strategy | First Warmup | Warmup Mean |
|---|---|---:|---:|
| CPU | Single DiT step | 593.83 ms | 401.22 ms |
| CPU | Python 4-step loop | 1215.16 ms | 1224.23 ms |
| CPU | Fused 4-step IR | 1167.59 ms | 971.47 ms |
| GPU | Single DiT step | 296.88 ms | 44.18 ms |
| GPU | Python 4-step loop | 64.20 ms | 64.56 ms |
| GPU | Fused 4-step IR | 85.46 ms | 57.40 ms |

## Interpretation

- The Intel Arc 140V iGPU is currently the viable deployment target for the DiT action head.
- The fused-loop graph is still faster than Python orchestration on both CPU and GPU, but the relative gain is smaller on the Intel GPU than on the earlier Apple CPU baseline.
- GPU fused-loop latency is approximately 54.39 ms per 4-step action chunk, or about 18.39 action chunks/sec under the deep synthetic benchmark.
- CPU latency is much higher on this target for the DiT action path: 874.07 ms for the fused graph.
- NPU was not benchmarked in this run because the NPU compiler previously aborted on the dynamic DiT graph during shape/type inference. It should be treated as a separate static-shape enablement task.

## Comparison to Earlier Local Baseline

| Platform | Strategy | Latency |
|---|---|---:|
| Apple M4 CPU | Python-orchestrated loop | 495.65 ms |
| Apple M4 CPU | Fused 4-step IR | 259.50 ms |
| Intel Core Ultra 7 258V CPU | Python-orchestrated loop | 1223.56 ms |
| Intel Core Ultra 7 258V CPU | Fused 4-step IR | 873.17 ms |
| Intel Arc 140V iGPU | Python-orchestrated loop | 65.31 ms |
| Intel Arc 140V iGPU | Fused 4-step IR | 54.39 ms |

The cross-platform CPU numbers are not directly comparable because the latest Intel run used G1 dimensions (`25 x 23` action chunks), while earlier local notes used LIBERO dimensions (`8 x 7`). The Intel GPU result should be treated as the first meaningful target-hardware baseline.

## VLM and Full VLA Status

The DiT action-head path is now benchmarked on Intel CPU/GPU. The full VLA path is not yet benchmarked end-to-end because the repository still does not contain a real Qwen2.5-VL OpenVINO export.

The existing `qwen_vlm_backbone.bin` in prior local artifacts was a 28-byte mock/template artifact, not real VLM weights. Therefore it is not valid to present it as a Qwen2.5-VL latency benchmark.

Next work for full VLA benchmarking:

1. Export the real Qwen2.5-VL backbone with official weights on the Intel machine.
2. Benchmark VLM prefill/feature extraction separately on GPU.
3. Measure tensor handoff overhead from VLM hidden states into the DiT action head.
4. Combine VLM latency plus fused DiT latency into a true full action-generation latency.

## Immediate Next Step

Start GPU profiling with VTune/OpenVINO profiling focused on:

- attention blocks,
- decomposed AdaLayerNorm/MVN patterns,
- matmul utilization on Arc 140V,
- fused-loop graph scheduling.

In parallel, unblock full VLA benchmarking by replacing the current mock/template `qwen_vlm_backbone` artifact with a real Qwen2.5-VL OpenVINO export.
