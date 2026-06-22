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

## Latency Results

| Device | Python-Orchestrated Single-Step Loop | Fused 4-Step IR | Fused Speedup |
|---|---:|---:|---:|
| CPU | 1215.52 ms | 874.07 ms | 1.39x |
| GPU | 69.02 ms | 57.87 ms | 1.19x |
| NPU | skipped | skipped | n/a |

## Interpretation

- The Intel Arc 140V iGPU is currently the viable deployment target for the DiT action head.
- The fused-loop graph is still faster than Python orchestration on both CPU and GPU, but the relative gain is smaller on the Intel GPU than on the earlier Apple CPU baseline.
- GPU fused-loop latency is approximately 57.87 ms per 4-step action chunk, or about 17.3 action chunks/sec under this synthetic benchmark.
- CPU latency is much higher on this target for the DiT action path: 874.07 ms for the fused graph.
- NPU was not benchmarked in this run because the NPU compiler previously aborted on the dynamic DiT graph during shape/type inference. It should be treated as a separate static-shape enablement task.

## Comparison to Earlier Local Baseline

| Platform | Strategy | Latency |
|---|---|---:|
| Apple M4 CPU | Python-orchestrated loop | 495.65 ms |
| Apple M4 CPU | Fused 4-step IR | 259.50 ms |
| Intel Core Ultra 7 258V CPU | Python-orchestrated loop | 1215.52 ms |
| Intel Core Ultra 7 258V CPU | Fused 4-step IR | 874.07 ms |
| Intel Arc 140V iGPU | Python-orchestrated loop | 69.02 ms |
| Intel Arc 140V iGPU | Fused 4-step IR | 57.87 ms |

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

Run the in-depth benchmark workflow to collect compile time, first-call latency, steady-state latency distribution, IR op counts, file sizes, and CPU/GPU stability metrics. After that, start GPU profiling with VTune/OpenVINO profiling focused on:

- attention blocks,
- decomposed AdaLayerNorm/MVN patterns,
- matmul utilization on Arc 140V,
- fused-loop graph scheduling.
