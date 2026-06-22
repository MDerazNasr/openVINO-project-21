# Presentation and Team Update Plan

## Purpose

This note turns the current project state into a mentor/team presentation plan.

It should be used together with:

- `notes/project_log_and_next_plan.md`
- `notes/mentor_update_email_2026_06_22.md`
- `notes/vtune_profiling_plan.md`
- `notes/flops_roofline_learning_plan.md`
- `notes/ir_graph_comparison.md`
- `notes/vlm_export_status.md`

The goal is to prepare a clear update that explains what we tried, why we tried it, what worked, what failed, and what the next technical decision depends on.

## Audience

There are two expected audiences:

1. Mentors directly involved in this project.
2. The wider team / group presentation audience.

The mentor version can be more technical and decision-focused.

The wider team version should explain the model path and benchmark story more carefully:

```text
VLM backbone
-> language/vision hidden states
-> DiT action head
-> denoising loop
-> robot action chunk
```

## Core Story

The project has moved from export feasibility into hardware benchmarking and profiling.

The story should be:

1. We identified the VLA inference path and export blockers.
2. We cleaned the DiT graph boundary enough to export to OpenVINO IR.
3. We validated numerical parity for the DiT action head.
4. We compared two deployment architectures:
   - Python-orchestrated single-step loop,
   - fused 4-step OpenVINO IR.
5. The fused graph preserved weight memory because OpenVINO shares/deduplicates weights across unrolled steps.
6. We benchmarked the DiT action head on Intel CPU/GPU hardware.
7. Intel Arc 140V gives strong absolute latency, and fused-loop remains faster than Python orchestration.
8. The next phase is VTune profiling to choose the first real optimization target.
9. Full VLA latency is still blocked until the real Qwen2.5-VL backbone is exported with weights.

Update from June 23, 2026:

- A deck-ready mentor presentation package has been drafted in `notes/mentor_presentation_package_2026_06_23.md`.
- A mentor email/update draft has been drafted in `notes/mentor_update_email_2026_06_23.md`.
- The presentation should use the latest Intel benchmark and OpenVINO profiling results, not the older local-only deck numbers.
- The presentation should explicitly say that the current validated benchmark is DiT action-head latency, while full VLA latency remains blocked on a real Qwen2.5-VL export with weights.
- The presentation should also explicitly say that VTune was not available on the runner and that OpenVINO node profiling was used as an interim fallback.

## Slide Outline

### Slide 1: Project Goal

Explain the GSoC objective:

```text
Optimize UnifoLM-VLA inference on Intel hardware using OpenVINO.
```

Include the high-level pipeline:

```text
image + instruction + robot state
-> Qwen2.5-VL / VLM
-> VLM embeddings
-> DiT action head
-> denoising loop
-> action chunk
```

### Slide 2: Proposal Timeline vs Current Status

Show that we are transitioning from:

```text
Weeks 3-4: iGPU baseline
```

to:

```text
Weeks 5-6: VTune profiling and bottleneck analysis
```

Status:

- DiT action-head export: complete.
- DiT Intel CPU/GPU benchmark: complete.
- Full VLA benchmark: blocked on real Qwen2.5-VL export.
- VTune profiling: next.

### Slide 3: Export Blockers and Fixes

Use a table:

| Blocker | Why It Was A Problem | What We Did |
|---|---|---|
| Python denoising loop | Repeated host/runtime calls | Tested single-step and fused-loop IR |
| `torch.randn` | Non-deterministic graph input | Externalized noise input |
| `torch.autocast` | Hidden precision policy | Moved precision concern outside graph |
| `BatchFeature` / Python containers | Breaks clean tensor boundary | Patched toward tensor/dict boundary |

Message:

```text
The main lesson was that export quality depends on graph boundaries, not just conversion syntax.
```

### Slide 4: Numerical Parity

Report:

```text
MSE: 0.000590%
MAE: 0.00196
```

Interpretation:

- MSE is the main structural check.
- MAE is not a serious blocker at this stage.
- Need to rerun parity with target precision/hardware as optimization continues.

Do not overclaim this as full VLA quality validation. It validates the DiT action-head graph path.

### Slide 5: Single-Step vs Fused-Loop Architecture

Show:

```text
Python loop:
  OpenVINO single-step call x4

Fused loop:
  one OpenVINO graph with 4 unrolled steps
```

Why this matters:

- Fused-loop reduces Python/runtime orchestration.
- It gives OpenVINO more graph to optimize.
- It is better aligned with deployment if denoising step count is fixed.

### Slide 6: Weight Sharing Result

Show the important artifact result:

```text
single_step_dit.bin ~= 1.1GB
fused_loop_dit.bin  ~= 1.1GB
```

Also show:

```text
single_step_dit.xml -> smaller
fused_loop_dit.xml  -> larger
```

Interpretation:

- XML graph complexity grows because compute is unrolled.
- `.bin` does not grow 4x because OpenVINO shares/deduplicates weights.
- This makes fused-loop viable.

### Slide 7: Intel Hardware

Report the benchmark machine:

```text
CPU: Intel Core Ultra 7 258V
GPU: Intel Arc 140V GPU (16GB) iGPU
NPU: Intel AI Boost
OpenVINO: 2026.2.1
```

Mention the control path:

```text
GitHub Actions self-hosted runner
```

Why:

- Direct SSH from Mac was blocked by missing jump-host credentials.
- GitHub runner let us execute reproducible workflows on the target hardware.

### Slide 8: DiT Hardware Benchmark Results

Use the validated table:

| Device | Single Step | Python 4-Step Loop | Fused 4-Step IR | Fused Speedup | Fused Throughput |
|---|---:|---:|---:|---:|---:|
| CPU | 307.54 ms | 1223.56 ms | 873.17 ms | 1.40x | 1.15 chunks/s |
| GPU | 16.20 ms | 65.31 ms | 54.39 ms | 1.20x | 18.39 chunks/s |
| NPU | n/a | n/a | skipped | n/a | n/a |

Message:

- Intel Arc 140V is the best current target.
- Fused-loop remains faster than Python orchestration.
- NPU is deferred because this dynamic DiT graph previously caused compiler issues.

### Slide 9: IR Graph Comparison

Show:

```text
single-step ops: 1,345
fused ops:       4,270
op ratio:        3.17x
BIN ratio:       ~1.00x
```

Important op scaling:

```text
ScaledDotProductAttention: 16 -> 64
MVN:                       33 -> 132
MatMul:                    123 -> 438
```

Interpretation:

- Compute graph is unrolled.
- Weight storage is shared.
- MVN/AdaLayerNorm patterns scale with denoising steps and may be a kernel-fusion candidate, but VTune must confirm.

### Slide 10: FLOPs / Roofline Learning

Explain the theory at a high level:

```text
T_math = FLOPs / hardware_compute_throughput
T_mem  = bytes_moved / memory_bandwidth
T_total ~= max(T_math, T_mem)
```

Then:

```text
Arithmetic intensity = FLOPs / bytes moved
```

Use this to explain:

- compute-bound means math is the bottleneck,
- communication/memory-bound means data movement is the bottleneck,
- roofline helps decide which optimization type is worth doing.

Connect to the project:

- DiT transformer matmuls/attention may be compute-heavy.
- AdaLayerNorm/MVN/elementwise patterns may be memory or dispatch-sensitive.
- VLM must also be analyzed once real export is available.

### Slide 11: VTune Profiling Plan

Explain VTune:

```text
VTune breaks a latency number into where time is actually spent.
```

Profile targets:

1. Python-orchestrated single-step loop.
2. Fused 4-step DiT IR.

Questions to answer:

- Are attention/matmul kernels dominant?
- Are MVN/AdaLayerNorm and elementwise kernels significant?
- Are there many small kernels / dispatch gaps?
- Is GPU utilization high or low?
- Is memory bandwidth a bottleneck?

Decision rule:

```text
Choose the first kernel/runtime contribution based on measured hotspots, not assumptions.
```

Current interim result:

```text
VTune is not installed/on PATH on the runner.
OpenVINO node profiling was added as a fallback.
Initial OpenVINO profile: MLP FullyConnected/MatMul dominates, self-attention projections are second, SDPA is visible, and MVN is much smaller.
```

Presentation caveat:

```text
This is not a VTune result and should be described as interim OpenVINO runtime profiling.
```

### Slide 12: Full VLA Status

Be explicit:

```text
Current latency numbers are DiT action-head numbers, not full VLA latency.
```

Why full VLA is blocked:

- The Qwen2.5-VL `.bin` artifact is not present.
- The existing VLM IR was a mock/template artifact, not real model weights.
- Benchmarking that would produce misleading numbers.

Next full VLA path:

1. Identify/load real Qwen2.5-VL checkpoint.
2. Export VLM backbone to real OpenVINO IR.
3. Benchmark VLM alone.
4. Benchmark VLM + fused DiT end-to-end.
5. Investigate zero-copy tensor handoff or combined graph.

### Slide 13: Pretrained Weights Clarification

State clearly:

```text
The current DiT benchmark validates OpenVINO export/runtime behavior for the configured DiT graph.
It should not be presented as a trained end-to-end robot policy benchmark unless the real trained weights and full VLA path are loaded.
```

What we should do next:

- Identify the exact pretrained checkpoint expected by UnifoLM-VLA.
- Confirm whether the DiT action head weights are loaded from pretrained files or initialized by config in each benchmark script.
- Add script logging that prints checkpoint path / missing checkpoint status.
- Separate labels in slides:
  - structural export benchmark,
  - trained DiT benchmark,
  - full VLA benchmark.

### Slide 14: Next Checklist

Prioritized:

1. Run VTune availability check on Intel runner.
2. Create VTune profiling target script for fused DiT and Python loop.
3. Collect GPU Hotspots traces.
4. Validate FLOPs / roofline calculations layer-by-layer.
5. Ask mentors which FLOPs tool/library they recommend.
6. Export real Qwen2.5-VL backbone.
7. Benchmark VLM and full VLA.
8. Decide first kernel/runtime contribution.
9. Prepare benchmark + presentation package and send to mentors.

## Questions To Ask Before Wider Team Presentation

Ask the organizer/mentor:

- When is the presentation?
- How long should it be?
- Is the audience mostly OpenVINO runtime engineers, GSoC students, or robotics/VLA researchers?
- Should the presentation focus on project progress, technical details, or blockers?
- Should live benchmark numbers be included, or only summarized?
- Should the deck include VTune results if available, or should those wait for a later update?
- Is the expected format slides, demo, written update, or all three?
- Should we include code-level OpenVINO IR details?
- Should we include open questions for the team, such as zero-copy handoff or VLM export approach?

## Questions To Ask Mentors

Technical:

- Which FLOPs counting tool/library did they have in mind?
- Should multiply-add be counted as 1 operation or 2 FLOPs for our roofline slide?
- Which Intel Arc 140V peak compute and bandwidth numbers should we use?
- Do they prefer VTune GPU Hotspots first or OpenVINO runtime profiling first?
- Should the first contribution target AdaLayerNorm/MVN only if VTune confirms it?
- Do they want us to attempt NPU static-shape enablement now or defer it?

Full VLA:

- Which exact pretrained UnifoLM-VLA / Qwen2.5-VL checkpoint should be used?
- Is there an internal/private checkpoint or only public weights?
- Should VLM and DiT remain separate with zero-copy handoff, or should we attempt a combined graph?
- What full VLA input shape and prompt/image setup should be used for benchmark comparability?

Presentation:

- Should we send the benchmark results and slides before the next meeting?
- Should we include the GitHub Actions artifact link?
- Should we include the failed attempts and blockers, or keep the presentation focused on final results?

## What Not To Claim Yet

Do not claim:

- full VLA end-to-end latency,
- real robot-policy quality,
- Qwen2.5-VL benchmark numbers,
- NPU support,
- final kernel bottleneck,
- final roofline conclusion for Intel GPU,
- that the benchmark definitely uses pretrained action-head weights unless verified.

Safe claims:

- DiT action-head OpenVINO export works.
- Fused 4-step OpenVINO IR runs on Intel CPU/GPU.
- Fused-loop is faster than Python orchestration on the tested Intel hardware.
- OpenVINO shares/deduplicates weights across the unrolled fused-loop graph.
- Full VLA latency is blocked on real VLM export.
- VTune profiling is the correct next step before kernel contribution.

## Materials To Prepare

- [ ] Final benchmark table.
- [ ] IR graph comparison table.
- [ ] Diagram of Python-loop vs fused-loop architecture.
- [ ] FLOPs/roofline explanation slide.
- [ ] VTune profiling plan slide.
- [ ] Full VLA blocker slide.
- [ ] Pretrained-weight clarification slide.
- [ ] Next steps slide.
- [ ] Mentor email with benchmark results and presentation attached/linked.

## Tonight's Practical Priority

If time is limited, finish in this order:

1. Documentation and checklist integrity.
2. Presentation outline.
3. VTune target/profiling setup.
4. FLOPs validation tooling.
5. Real VLM export investigation.

Reason:

The presentation and mentor update need a coherent story first. VTune and full VLA work are the next technical milestones, but they should not block documenting what has already been proven.
