# Intel Hardware Full VLA + VTune Update - 2026-07-01

## Goal

Complete the next hardware milestone from the mentor checklist:

1. Get a real VLM benchmark instead of the earlier mock/tiny VLM artifact.
2. Connect the real VLM OpenVINO output to the fused DiT action head.
3. Run a true OpenVINO model-chain VLA benchmark on Intel CPU/GPU hardware.
4. Get VTune working on the Intel runner and collect a GPU Hotspots profile for the end-to-end VLA path.

This directly addresses mentor feedback that DiT alone may not be the full-model bottleneck and that the next optimization target should come from full VLA profiling, not from assumptions.

## Starting Problem

Earlier benchmark runs only measured the DiT action head with synthetic VLM embeddings.

That was useful for validating:

- DiT OpenVINO export.
- Single-step vs fused-loop deployment strategy.
- CPU/GPU hardware execution.
- Weight sharing across fused-loop unrolling.

But it was not enough for the next phase because the full VLA path also includes the Qwen/UnifoLM VLM backbone. The mentor specifically asked us to fix the VLM issue and benchmark the full model because the VLM may dominate runtime.

## Why The VLM Was Previously Missing

The repo contained a tiny `qwen_vlm_backbone.xml`, but no real matching `.bin` weights.

That meant:

- It could not represent the real Qwen/UnifoLM VLM backbone.
- It could not be used for honest VLM latency.
- It could not support full end-to-end VLA latency claims.

We intentionally excluded it from reported latency because using it would have produced misleading results.

## What We Changed

### Real VLM Export Route

Direct PyTorch-to-OpenVINO conversion of the Qwen2.5-VL visual path hit unsupported frontend patterns around visual indexing/windowing.

Instead of faking the result, we used an ONNX bridge:

```text
UnifoLM VLM checkpoint
-> processor-generated multimodal example inputs
-> ONNX export
-> OpenVINO IR conversion
-> OpenVINO VLM benchmark
```

The real checkpoint used was:

```text
unitreerobotics/Unifolm-VLM-Base
```

### Persisting The Huge VLM IR

The real VLM OpenVINO `.bin` is about 15.5 GB.

Uploading that through GitHub Actions artifacts was impractical and previously left the self-hosted runner stuck during artifact upload.

Decision:

- Do not upload the VLM `.bin`.
- Persist it on the Windows runner disk.
- Upload only small benchmark reports.

Persistent runner path:

```text
C:\Users\devcloud\openVINO-project-21-main\artifacts\openvino_ir
```

This lets later hardware benchmark jobs copy the real VLM IR into their clean GitHub Actions workspace without transferring the 15.5 GB weight file through GitHub artifact storage.

## VLM ONNX Export Check Run

Workflow:

```text
VLM ONNX Export Check
```

Run:

```text
https://github.com/MDerazNasr/openVINO-project-21/actions/runs/28499430284
```

Result:

```text
success
```

Important steps that passed:

- Real checkpoint load.
- ONNX export.
- ONNX-to-OpenVINO conversion.
- OpenVINO IR inspection.
- Persist OpenVINO VLM IR on runner.
- VLM-only OpenVINO benchmark.
- VLM-compatible DiT IR generation.
- End-to-end OpenVINO VLA benchmark.

Generated real VLM IR:

| Artifact | Size |
|---|---:|
| `qwen_vlm_backbone_from_onnx.xml` | 4,844,170 bytes |
| `qwen_vlm_backbone_from_onnx.bin` | 15,494,478,376 bytes |

VLM input shapes:

| Input | Shape | Type |
|---|---:|---|
| `input_ids` | `[1, 92]` | `int64` |
| `attention_mask` | `[1, 92]` | `int64` |
| pixel tensor | `[256, 1176]` | `float32` |

VLM output shape:

```text
[1, 92, 3584]
```

## VLM-Only Hardware Benchmark

From run `28499430284`:

| Device | Mean | Median | P95 | Compile | Status |
|---|---:|---:|---:|---:|---|
| CPU | 26052.15 ms | 22745.36 ms | 58996.57 ms | 24623.14 ms | ok |
| GPU | 214.86 ms | 207.56 ms | 266.20 ms | 54055.01 ms | ok |
| NPU | n/a | n/a | n/a | n/a | skipped |

Interpretation:

- The real VLM is much heavier than the DiT action head on GPU.
- This supports the mentor's concern that DiT alone may not be the main optimization target.
- CPU VLM latency is very large and variable; GPU is the meaningful deployment baseline for now.

## End-to-End OpenVINO VLA Benchmark

This benchmark measures:

```text
VLM OpenVINO IR inference
-> Python tensor handoff
-> fused DiT OpenVINO IR inference
-> action chunk output
```

It does not include Qwen processor/image preprocessing.

From run `28499430284`:

| Device | VLM Mean | DiT Mean | End-to-End Mean | End-to-End P95 | VLM Output | DiT Output | Status |
|---|---:|---:|---:|---:|---|---|---|
| CPU | 46504.76 ms | 936.17 ms | 73189.04 ms | 74037.18 ms | `[1, 92, 3584]` | `[1, 25, 23]` | ok |
| GPU | 204.49 ms | 54.86 ms | 276.91 ms | 278.27 ms | `[1, 92, 3584]` | `[1, 25, 23]` | ok |
| NPU | n/a | n/a | n/a | n/a | n/a | n/a | skipped |

Follow-up hardware workflow run `28502030348` reproduced the same model-chain result:

| Device | VLM Mean | DiT Mean | End-to-End Mean | End-to-End P95 | Status |
|---|---:|---:|---:|---:|---|
| CPU | 71021.30 ms | 2000.07 ms | 67203.91 ms | 67203.91 ms | ok |
| GPU | 206.13 ms | 54.77 ms | 278.90 ms | 279.70 ms | ok |
| NPU | n/a | n/a | n/a | n/a | skipped |

The CPU numbers vary significantly because the VLM path is very slow on CPU. The GPU e2e number is stable around `277-279 ms`.

## VTune Setup

Earlier state:

- VTune harness existed.
- VTune was not available on runner `PATH`.
- We used OpenVINO `PERF_COUNT=YES` node profiling as a fallback.

What changed:

- Added a VTune diagnostic workflow.
- Confirmed VTune was initially missing.
- Installed Intel oneAPI VTune Profiler through the runner setup path.
- Resolved VTune from the known oneAPI install path:

```text
C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe
```

Installed version:

```text
Intel(R) VTune(TM) Profiler 2025.3.0
```

We first validated VTune with a small smoke test before profiling the full VLA path. That avoided wasting time debugging full-model issues when the profiler itself might still be broken.

## Invalid VTune Attempt And Fix

The first end-to-end VTune run was not valid because the hardware benchmark workflow could not find:

```text
qwen_vlm_backbone_from_onnx.xml
qwen_vlm_backbone_from_onnx.bin
```

The VTune step still produced a result directory, but it had only profiled a short failing Python process. We do not treat that as a real VLA profile.

Root cause:

- The VLM ONNX workflow generated the huge VLM IR in its clean GitHub Actions workspace.
- That clean workspace was deleted after the job.
- The later hardware benchmark job could not see the generated VLM IR.

Fix:

- Persist the real VLM IR to the runner disk.
- Re-run the VLM workflow to populate the persistent path.
- Re-run the hardware benchmark with `run_vtune=true` and `vtune_target=e2e`.

## Valid VTune E2E Run

Workflow:

```text
Intel Hardware Benchmark
```

Run:

```text
https://github.com/MDerazNasr/openVINO-project-21/actions/runs/28502030348
```

Result:

```text
success
```

VTune target:

```text
e2e
```

VTune command profiled:

```text
python export_tests\profile_vla_workload.py --device GPU --mode e2e --iterations 2 --warmup 2 --output-json benchmark_outputs\vla_profile_e2e.json
```

Produced files:

- `vla_profile_e2e.json`
- `vtune_gpu_vla_e2e_summary.txt`
- Full `vtune_gpu_vla_e2e/` result directory

Profile workload latency:

| Device | Mode | Mean | Min | Max | Iterations |
|---|---|---:|---:|---:|---:|
| GPU | e2e | 275.54 ms | 275.24 ms | 275.84 ms | 2 |

The profiled workload used the real VLM IR and fused DiT IR:

```text
VLM output shape: [1, 92, 3584]
```

## VTune Findings

VTune summary:

| Metric | Value |
|---|---:|
| Elapsed time | 39.670 s |
| GPU time | 1.787 s |
| XVE Array stalled/idle | 88.0% of elapsed time with GPU busy |
| Occupancy | 66.6% of peak |

Hottest GPU tasks listed by VTune:

| Task | Total Time | SIMD Width | SIMD Utilization |
|---|---:|---:|---:|
| `gemm_kernel` | 0.130 s | 16 | 100.0% |
| `gemm_kernel` | 0.084 s | 16 | 100.0% |
| `gemm_kernel` | 0.056 s | 16 | 100.0% |
| `[Others]` | 0.378 s | n/a | 0.0% |

VTune recommendations included:

- GPU utilization is low relative to elapsed time.
- XVE stalled/idle is high.
- Execution time on device is less than memory transfer time.

Interpretation:

- The full VLA chain is not simply "compute everything faster" on GPU.
- Host orchestration, model handoff, memory movement, and offload efficiency are now important to inspect.
- This strengthens the case for investigating zero-copy/shared tensor handoff between VLM output and DiT input.
- The hottest named kernels are GEMM-like, consistent with the OpenVINO node profile showing large FullyConnected/MatMul contribution.

## OpenVINO Node Profiling Cross-Check

The OpenVINO fallback profile is still useful because it maps runtime to graph-level operator categories.

Latest fused DiT GPU node profile:

| Category | Share |
|---|---:|
| MLP FullyConnected | 54.73% |
| Self-attention projection | 25.63% |
| Other FullyConnected | 12.25% |
| Normalization MVN | 2.21% |
| Attention SDPA | 1.98% |

Interpretation:

- DiT runtime remains dominated by FullyConnected/MatMul-style work.
- MVN/AdaLayerNorm is measurable but not the dominant category in this profile.
- A first optimization target should be chosen using full VLA profiling, not only the original proposal guess.

## What This Means For The Project

We have now moved from:

```text
DiT-only benchmark with synthetic VLM embeddings
```

to:

```text
real VLM OpenVINO export
-> VLM-only hardware benchmark
-> VLM-compatible DiT benchmark
-> OpenVINO model-chain VLA benchmark
-> real VTune GPU Hotspots profile for e2e VLA workload
```

This is a major milestone for the proposal's benchmarking/profiling phase.

The main remaining caveat:

- This is an OpenVINO model-chain benchmark.
- It does not include Qwen processor/image preprocessing.
- The VLM-to-DiT handoff is currently through Python.

That caveat is now a concrete optimization opportunity rather than a blocker.

## Next Work

1. Analyze the full VTune result directory in the VTune GUI or with more detailed CLI reports.
2. Run separate VTune profiles for:
   - `vlm`
   - `fused_dit`
   - `e2e`
3. Compare where time is spent:
   - VLM kernels
   - DiT kernels
   - Python/handoff overhead
   - memory transfer/offload overhead
4. Investigate static-shape compilation everywhere possible, because mentors noted dynamic shapes can force fallback kernels.
5. Investigate zero-copy or shared tensor handoff between VLM output and DiT input.
6. Decide first optimization target based on combined evidence:
   - VTune e2e profile
   - OpenVINO node profile
   - FLOPs/roofline analysis
   - mentor feedback

