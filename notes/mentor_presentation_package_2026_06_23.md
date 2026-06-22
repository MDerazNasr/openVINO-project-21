# Mentor Presentation Package - June 23, 2026

## Purpose

This is the deck-ready narrative for the next mentor/team update. It updates the previous slide story with the real Intel hardware benchmark results, the VTune availability blocker, the OpenVINO profiling fallback, and the current full-VLA limitation.

Use this with:

- `notes/project_log_and_next_plan.md`
- `notes/openvino_node_profile_2026_06_23.md`
- `notes/vtune_profiling_plan.md`
- `notes/flops_roofline_learning_plan.md`
- `notes/vlm_export_status.md`
- `notes/ir_graph_comparison.md`

The key correction from the earlier deck is that we should no longer present the Qwen2.5-VL path as a completed real full-VLA benchmark. The current validated benchmark is the DiT action head with synthetic VLM embeddings.

## One-Sentence Update

We now have a reproducible Intel hardware benchmark for the OpenVINO DiT action head: the fused 4-step graph runs at about `54.48 ms` per action chunk on Intel Arc 140V, is `1.17x` faster than Python loop orchestration, preserves the single-step weight footprint, and initial OpenVINO profiling shows matmul/FullyConnected work dominates while real VLM/full-VLA latency remains blocked on exporting Qwen2.5-VL with actual weights.

## What To Change From The Previous Deck

Keep:

- The explanation of graph-boundary cleanup.
- The deterministic parity/root-cause story.
- The single-step vs fused-loop comparison.
- The weight-sharing result.
- The idea that hardware profiling is the next phase.

Replace or soften:

- Replace local CPU-only fused speedup numbers with Intel CPU/GPU results.
- Replace "Qwen2.5-VL exported" with "Qwen2.5-VL full export is still blocked by missing real `.bin` weights."
- Replace "VTune profiling completed" with "VTune harness added; VTune unavailable on runner; OpenVINO node profiling added as fallback."
- Replace "AdaLayerNorm/MVN is the bottleneck" with "MVN is measurable, but current profile is dominated by FullyConnected/matmul work; first kernel target should be decided after deeper profiling."
- Avoid saying this is full end-to-end VLA latency.

## Slide Draft

### Slide 1 - Title

Title:

```text
UnifoLM-VLA OpenVINO Optimization
Intel Hardware Baseline and Profiling Direction
```

Subtitle:

```text
GSoC 2026 - OpenVINO Project 21
June 2026 update
```

Speaker note:

```text
This update is about moving from export feasibility into measured Intel hardware performance. The main validated result is currently the DiT action head, not the full VLA pipeline yet.
```

### Slide 2 - Current Phase

Main message:

```text
We have completed the DiT action-head Intel baseline and are entering the profiling/bottleneck-analysis phase.
```

Table:

| Area | Status |
|---|---|
| DiT single-step OpenVINO export | Complete |
| DiT fused-loop OpenVINO export | Complete |
| Intel CPU/GPU DiT benchmark | Complete |
| NPU benchmark | Deferred due dynamic graph/compiler risk |
| VTune profiling | Harness added, tool unavailable on runner |
| OpenVINO node profiling fallback | Complete |
| Real Qwen2.5-VL / full VLA benchmark | Blocked on real VLM export weights |

Speaker note:

```text
This lines up with the proposal transition from iGPU baseline work into profiling. The important distinction is that the DiT benchmark is real and hardware-backed, while full VLA remains a next-step blocker.
```

### Slide 3 - Model Path And Benchmark Scope

Pipeline:

```text
image + language instruction + robot state
-> Qwen2.5-VL / VLM backbone
-> VLM hidden states
-> DiT action head
-> denoising loop
-> action chunk
```

Validated benchmark scope:

```text
synthetic VLM hidden states + robot state
-> DiT action head
-> 4-step denoising
-> action chunk
```

Not yet validated:

```text
real image/text input
-> real Qwen2.5-VL OpenVINO backbone
-> DiT action head
-> full end-to-end VLA latency
```

Speaker note:

```text
The benchmark isolates the action-head path. That is still useful because the DiT denoising loop is repeated and latency-critical, but it is not the same as full VLA latency.
```

### Slide 4 - Export Blockers Resolved For The DiT Path

| Blocker | Why It Mattered | Current Resolution |
|---|---|---|
| Python denoising loop | Repeated Python/runtime boundary | Compared Python loop vs fused OpenVINO loop |
| `torch.randn` | Non-deterministic graph input | Externalized noise |
| `torch.autocast` | Hidden precision policy | Moved precision outside graph boundary |
| `BatchFeature` / Python objects | Poor tensor graph boundary | Patched toward tensor/dict boundary |
| Hardcoded local paths | Broke on Intel runner | Converted scripts to repo-relative paths |

Speaker note:

```text
The main lesson is that OpenVINO conversion is not only about calling the converter. It depends heavily on giving the compiler a clean tensor boundary.
```

### Slide 5 - Numerical Parity

Result from deterministic DiT validation:

```text
MSE: 0.000590%
MAE: 0.00196
```

Interpretation:

- MSE passes the structural conversion check.
- Mentor feedback: MAE is not a severe issue at this stage.
- This validates the DiT graph path, not trained full-VLA policy quality.
- Parity should be rerun when real checkpoints and target precision settings are finalized.

Speaker note:

```text
The key fix was enforcing deterministic initialization and inputs. Before that, the comparison could mix conversion error with random baseline drift.
```

### Slide 6 - Pretrained Weights Clarification

Current benchmark:

```text
DiT architecture/config benchmark using generated or available model weights.
```

What this means:

- Valid for latency, export structure, graph size, memory footprint, and OpenVINO execution behavior.
- Not yet valid as a trained-policy quality or accuracy result unless we confirm and load the official pretrained action-head checkpoint.

Next action:

```text
Identify the official pretrained UnifoLM-VLA checkpoints and clearly label each run as:
1. structural synthetic benchmark,
2. trained checkpoint benchmark,
3. full VLA benchmark.
```

Speaker note:

```text
The mentor specifically asked whether pretrained weights were used. For presentation safety, we should separate architecture/runtime benchmarking from trained-policy benchmarking.
```

### Slide 7 - Single-Step vs Fused-Loop Deployment

Python loop:

```text
for 4 denoising steps:
    call OpenVINO single-step DiT
```

Fused OpenVINO loop:

```text
one OpenVINO graph
with 4 unrolled denoising steps
```

Why this matters:

- Fewer Python/runtime calls.
- More graph available to OpenVINO transformations.
- Better deployment candidate when denoising step count is fixed.

Speaker note:

```text
The fused graph is not just syntactic cleanup. It changes the runtime architecture by moving orchestration into the compiled graph.
```

### Slide 8 - Weight Sharing Result

Observed artifact sizes:

| Artifact | Size Behavior |
|---|---|
| `single_step_dit.bin` | about `1.1236 GB` |
| `fused_loop_dit.bin` | about `1.1236 GB` |
| `single_step_dit.xml` | smaller graph |
| `fused_loop_dit.xml` | larger graph |

IR comparison:

| Metric | Single Step | Fused 4-Step | Ratio |
|---|---:|---:|---:|
| Ops | 1,345 | 4,270 | 3.17x |
| Weight file | ~1.12 GB | ~1.12 GB | ~1.00x |

Speaker note:

```text
This was the most important deployment result: graph complexity increases, but the weight file does not grow four times. OpenVINO shares/deduplicates the repeated weights.
```

### Slide 9 - Intel Hardware Baseline

Hardware:

```text
CPU: Intel Core Ultra 7 258V
GPU: Intel Arc 140V GPU (16GB) iGPU
NPU: Intel AI Boost
OpenVINO: 2026.2.1
```

Execution setup:

```text
GitHub Actions self-hosted runner on the Intel machine
```

Why this setup:

- Direct SSH from the Mac was blocked by missing jump-host credentials.
- GitHub runner gave a reproducible path to run code on the target hardware.
- Results are preserved in GitHub Actions logs and artifacts.

### Slide 10 - DiT Hardware Benchmark Results

Latest benchmark:

| Device | Single Step Mean | Python Loop Mean | Fused Loop Mean | Fused Speedup | Fused Chunks/s |
|---|---:|---:|---:|---:|---:|
| CPU | 306.02 ms | 1253.17 ms | 869.47 ms | 1.44x | 1.15 |
| GPU | 15.93 ms | 63.88 ms | 54.48 ms | 1.17x | 18.36 |
| NPU | n/a | n/a | n/a | n/a | skipped |

Interpretation:

- Intel Arc 140V is the strongest current target.
- Fused-loop is still faster than Python orchestration.
- GPU absolute latency is much better than CPU.
- NPU remains opt-in/deferred until the dynamic graph issue is handled safely.

Speaker note:

```text
The GPU fused speedup is smaller than the CPU speedup, but the absolute GPU latency is the deployment-relevant result.
```

### Slide 11 - VTune Profiling Status

What VTune would answer:

```text
During the 54 ms fused DiT GPU run, where is time spent?
Is the workload compute-bound, memory-bound, dispatch-bound, or blocked on specific kernels?
```

What happened:

- Added VTune workload harness.
- Added opt-in workflow input for VTune.
- Checked the Intel runner.
- VTune CLI was not found on `PATH`.

Decision:

```text
Use OpenVINO runtime node profiling as an interim hotspot view, while keeping VTune setup as the proper next profiling task.
```

Speaker note:

```text
We should not claim VTune results yet. We can claim that the VTune path is prepared and that OpenVINO profiling is our current fallback.
```

### Slide 12 - OpenVINO Node Profiling Fallback

GPU fused-loop profile category breakdown:

| Category | Share |
|---|---:|
| MLP FullyConnected | 52.15% |
| Self-attention projections | 26.01% |
| Other FullyConnected | 11.63% |
| SDPA attention | 4.42% |
| MVN / normalization | 2.21% |
| Layout/data movement | 1.24% |

Interpretation:

- MatMul/FullyConnected work dominates.
- Attention projection layers are the second-largest bucket.
- SDPA is visible but smaller.
- MVN/AdaLayerNorm is measurable, but not the top bottleneck in this profile.

Speaker note:

```text
This changes the kernel-fusion conversation. MVN may still be a contribution candidate, but current evidence says we should choose the first target after deeper profiling, not by assumption.
```

### Slide 13 - FLOPs And Roofline Learning

Definitions:

```text
FLOPs = floating-point operations
T_math = FLOPs / hardware compute throughput
T_mem = bytes moved / memory bandwidth
Arithmetic intensity = FLOPs / bytes moved
Runtime lower bound ~= max(T_math, T_mem)
```

How this applies here:

- Transformer MLP and attention projections are matmul-heavy and likely compute-heavy.
- MVN/AdaLayerNorm and elementwise chains can be memory or dispatch sensitive.
- Roofline helps decide whether to optimize math kernels, memory movement, graph boundaries, or dispatch overhead.

Next learning task:

```text
Validate FLOPs layer-by-layer instead of treating AI-generated FLOP totals as a black box.
```

Speaker note:

```text
The mentor specifically wanted a deeper understanding of how FLOPs are calculated, not just a final number. The plan is to review formulas per layer and compare them to profiler evidence.
```

### Slide 14 - VLM And Full VLA Status

Current blocker:

```text
No real Qwen2.5-VL OpenVINO .bin artifact is present.
```

What we intentionally did:

- Treated the tiny/template VLM artifact as mock.
- Excluded it from real latency reporting.
- Documented that full VLA latency is blocked.

What is needed for full VLA benchmarking:

1. Export real Qwen2.5-VL / VLM backbone with weights.
2. Validate its inputs/outputs and numerical sanity.
3. Benchmark VLM latency separately.
4. Connect VLM embeddings to DiT action head.
5. Measure full end-to-end latency.
6. Investigate zero-copy/shared tensor path to avoid host copies.

Speaker note:

```text
This is why we cannot report end-to-end VLA latency yet. The missing piece is not the DiT action head; it is the real VLM export with weights.
```

### Slide 15 - Decisions Needed From Mentors

Questions:

1. Should we prioritize installing/finding VTune on the runner, or continue with OpenVINO node profiling first?
2. Given the current profile, should the first optimization target remain AdaLayerNorm/MVN, or should we investigate FullyConnected/matmul/attention projection paths first?
3. Which official pretrained UnifoLM-VLA checkpoint should be used for trained-policy benchmarks?
4. What exact input shape/prompt/image setup should define the full VLA benchmark?
5. For the wider team presentation, what level of FLOPs/roofline detail is expected?

Speaker note:

```text
These are decision points, not blockers to all work. We can continue documenting, profiling, and preparing the VLM export path while these get clarified.
```

### Slide 16 - Next Two-Week Plan

Priority order:

1. Preserve current Intel benchmark and profiling artifacts.
2. Prepare mentor/team presentation from validated results.
3. Ask mentor about VTune setup and first kernel target.
4. Validate FLOPs/roofline math layer-by-layer.
5. Identify/load official pretrained checkpoints.
6. Export real Qwen2.5-VL with weights.
7. Benchmark VLM-only and full VLA latency.
8. Investigate zero-copy VLM-to-DiT handoff.
9. Select first kernel/runtime optimization based on profiling evidence.

Speaker note:

```text
This is enough work for a strong two-week update because it includes hardware results, profiling, theory validation, VLM blocker resolution, and a concrete path toward the first optimization contribution.
```

## Team Presentation Notes

For a wider OpenVINO/team audience, reduce project-specific detail and emphasize:

- The model structure: VLM plus DiT action head.
- Why diffusion/flow-matching action generation creates repeated compute.
- Why graph boundaries matter for OpenVINO.
- The surprising weight-sharing result.
- The Intel Arc 140V benchmark result.
- The profiling lesson: measure before choosing the first kernel.

Avoid spending too much time on:

- GitHub runner setup details.
- Path fixes.
- Failed artifact-copy attempts.
- Exact traceback history.

Mention those only if asked.

## Presentation-Safe Claims

Safe:

```text
The DiT action-head path exports and runs on Intel CPU/GPU with OpenVINO.
```

Safe:

```text
The fused 4-step OpenVINO graph is faster than Python loop orchestration on both CPU and GPU.
```

Safe:

```text
OpenVINO preserved the fused-loop weight footprint through weight sharing/deduplication.
```

Safe:

```text
OpenVINO node profiling suggests the current GPU DiT path is dominated by FullyConnected/matmul categories.
```

Not safe yet:

```text
We have full end-to-end VLA latency.
```

Not safe yet:

```text
VTune proves the kernel bottleneck.
```

Not safe yet:

```text
AdaLayerNorm/MVN is definitely the first kernel contribution.
```

Not safe yet:

```text
The benchmark proves trained robot policy quality.
```

## Open Items Before Final Slides

- [ ] Confirm whether the old deck should be edited directly or rebuilt as a new deck.
- [ ] Add screenshots/tables from GitHub Actions logs or benchmark artifacts.
- [ ] Decide whether to include the full profiling category table or only the top-five chart.
- [ ] Ask mentor/team when the wider presentation is happening.
- [ ] Ask what the team expects: high-level project update, kernel deep dive, or theory/roofline explanation.
- [ ] Ask mentor for the FLOPs library/tool they mentioned, or select a current PyTorch FLOP counter and validate formulas manually.
- [ ] Confirm official pretrained checkpoint source.
- [ ] Confirm whether VTune can be installed on the reserved Intel machine.
