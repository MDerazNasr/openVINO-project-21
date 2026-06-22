# Project Log and Next Plan

## Purpose

This note is the living record for the OpenVINO GSoC UnifoLM-VLA optimization project. It tracks what we tried, why we tried it, what worked, what failed, what decisions were made, and what needs to happen next.

The goal is to make the project easy to explain to mentors and to reuse this material later for slides, email updates, final reports, and technical handoff.

## Project Goal

The project aims to optimize a Vision-Language-Action model for embodied intelligence using OpenVINO.

The target architecture is:

```text
image + language instruction + robot state
-> Qwen2.5-VL / VLM backbone
-> VLM hidden states
-> flow-matching DiT action head
-> iterative denoising
-> robot action chunk
```

The proposal has three major technical deliverables:

1. Export the VLA model to OpenVINO IR.
2. Benchmark and profile it on Intel hardware.
3. Contribute kernel-level or runtime optimizations based on measured bottlenecks.

The stretch goal is to move more VLA-style action generation into a cleaner OpenVINO/OpenVINO GenAI runtime abstraction.

## Current Phase

According to the proposal timeline, we are now transitioning from:

```text
Weeks 3-4: iGPU baseline
```

into:

```text
Weeks 5-6: VTune profiling and bottleneck analysis
```

The DiT action-head iGPU baseline is now complete. Full VLA benchmarking is not complete yet because the real Qwen2.5-VL backbone has not been exported with actual weights.

Update from June 23, 2026:

- The VTune profiling harness and opt-in GitHub Actions workflow path are implemented.
- The normal Intel benchmark still passes after adding the profiling harness.
- VTune is not currently available on the Intel runner `PATH`; see `notes/intel_runner_vtune_check_2026_06_23.md`.
- OpenVINO runtime profiling was added as an interim hotspot map; see `notes/openvino_node_profile_2026_06_23.md`.
- Initial OpenVINO node profile suggests the DiT GPU path is dominated by MLP `FullyConnected` / MatMul work, followed by self-attention projections. SDPA is visible, and MVN is measurable but much smaller.
- Next profiling choice: locate/install VTune, or deepen the OpenVINO profile analysis by separating attention projections from MLP layers.

## What We Have Done

### 1. Runtime and Architecture Study

We first studied OpenVINO as a compiler/runtime stack rather than treating it as a black-box conversion library.

The working mental model:

```text
PyTorch model
-> OpenVINO frontend conversion
-> OpenVINO IR (.xml + .bin)
-> graph transformations and fusions
-> device plugin compile
-> hardware-specific kernel dispatch
```

This mattered because the VLA action model is not a single forward pass. It is closer to a diffusion/image-generation pipeline:

```text
initialize noise
-> repeatedly denoise/refine
-> produce final output
```

For UnifoLM-VLA:

```text
initialize noisy actions
-> repeatedly run DiT action head
-> integrate velocity updates
-> produce action chunk
```

This framing helped identify why graph boundaries, loop placement, and runtime dispatch overhead matter.

### 2. Inference Path Tracing

We traced the main inference path:

```text
LIBERO / robot inference wrapper
-> Unifolm_VLA.predict_action
-> Qwen2.5-VL backbone
-> last_hidden_state / VLM embeddings
-> FlowmatchingActionHead.predict_action
-> DiT denoising loop
-> action chunk
```

The important tensor interface into the DiT action head is:

```text
vl_embs: VLM hidden states
actions/noise: current action trajectory
state: proprioceptive robot state
timestep: denoising time
```

We confirmed that the repeated DiT path is the latency-critical section for the action head.

### 3. Export Blockers

The proposal identified four export blockers:

| Blocker | Problem | Resolution |
|---|---|---|
| Python denoising loop | Tracing full loop can unroll repeated DiT calls into static graph logic | Tested both single-step export and fused-loop export |
| `torch.randn` inside inference | Random generation inside graph boundary hurts deterministic export | Externalized noise input |
| `torch.autocast` in model path | Precision policy embedded in model code | Moved precision concerns to explicit compile/runtime settings |
| HuggingFace `BatchFeature` boundary | Python container is not a clean tensor graph input | Patched toward tensor/dict boundaries |

The main conclusion: the core DiT transformer compute is exportable once the graph boundary is clean.

### 4. Numerical Parity

We built a deterministic validation path by enforcing strict seeding.

Important result:

```text
MSE: 0.000590%
MAE: 0.00196
```

Mentor feedback:

- MSE passing is the important structural signal.
- MAE missing the strict target is not a serious blocker at this stage.
- MAE should be rerun on target hardware and with relevant precision settings.

Interpretation:

The graph topology appears structurally correct. Remaining drift is likely precision/runtime related rather than a conversion failure.

### 5. Native Source Patching

We did not rely only on fragile external wrappers. We also patched the model source to make the export boundary cleaner.

Why this decision was made:

- External wrappers are useful for quick experiments but hide the real model boundary.
- Native source patching makes the model more export-first.
- Mentors encouraged pushing graph boundaries deeper into the model so OpenVINO can optimize more of the math graph.

This direction was approved by mentors in the transcript.

### 6. Single-Step DiT Export

We first exported a single DiT denoising step.

Why:

- It is the smallest stable compute unit.
- It avoids Python loop/control-flow issues.
- It allows flexible denoising step counts.
- It is useful for debugging, parity, and profiling one repeated unit.

Result:

- Single-step DiT exported to OpenVINO IR.
- Runtime execution succeeded.
- Numerical parity was acceptable.

Decision:

Keep single-step IR as a research/debugging artifact, but do not assume it is the best deployment graph.

### 7. Fused 4-Step DiT Export

Mentors suggested fusing as much compute as possible into a larger graph so OpenVINO can optimize scheduling, layout, and kernel dispatch.

We tested a fused-loop wrapper that unrolls the 4-step denoising loop into one static OpenVINO graph.

Initial concern:

```text
4 denoising steps x 1.1GB DiT weights = possible 4.4GB artifact
```

Actual result:

```text
single_step_dit.bin ~= 1.1GB
fused_loop_dit.bin ~= 1.1GB
```

The XML graph grows, but the weight file stays flat. OpenVINO deduplicates/shares weights across unrolled steps.

Decision:

The fused-loop graph is the primary deployment candidate for fixed-step inference. The single-step graph remains useful when variable timestep count or debugging matters.

### 8. Intel Hardware Benchmark

We set up a GitHub Actions self-hosted runner on the Intel machine because direct SSH from the Mac was blocked by missing jump-host credentials.

The control path became:

```text
Mac edits code
-> git push
-> GitHub Actions
-> self-hosted Intel runner executes benchmark
-> logs/artifacts returned through GitHub
```

Hardware:

```text
CPU: Intel Core Ultra 7 258V
GPU: Intel Arc 140V GPU (16GB) iGPU
NPU: Intel AI Boost
OpenVINO: 2026.2.1
```

Deep benchmark result:

| Device | Single Step | Python 4-Step Loop | Fused 4-Step IR | Fused Speedup | Fused Throughput |
|---|---:|---:|---:|---:|---:|
| CPU | 307.54 ms | 1223.56 ms | 873.17 ms | 1.40x | 1.15 chunks/s |
| GPU | 16.20 ms | 65.31 ms | 54.39 ms | 1.20x | 18.39 chunks/s |
| NPU | n/a | n/a | skipped | n/a | n/a |

Interpretation:

- Intel Arc 140V GPU is the viable target for the DiT action head.
- Fused-loop remains faster than Python orchestration.
- GPU speedup from fusing is smaller than local CPU speedup, but absolute latency is much better.
- NPU is deferred because the dynamic DiT graph previously triggered a compiler abort.

### 9. Full VLA / VLM Status

Current benchmark is DiT-only:

```text
synthetic vl_embs
-> DiT action head
-> action chunk
```

It is not full VLA:

```text
image + text + state
-> Qwen2.5-VL
-> vl_embs
-> DiT
-> action chunk
```

Reason:

The repository does not currently contain a real Qwen2.5-VL OpenVINO export. The `qwen_vlm_backbone` artifact is missing/tiny/mock, so it cannot be used to report real VLM latency.

Decision:

Do not claim end-to-end VLA latency until the real Qwen2.5-VL backbone is exported with real weights.

## Pretrained Weights Clarification

The mentor asked whether the model was using pretrained weights or random initialization.

Current understanding:

- The DiT latency benchmark instantiates the action head architecture from config.
- This gives a valid structural/runtime benchmark because latency mostly depends on architecture, tensor shapes, and operators.
- However, unless a real trained action-head checkpoint is loaded, this should not be described as a trained-policy benchmark.

What is valid to claim now:

- OpenVINO can export and run the DiT graph.
- The fused-loop graph is faster than Python orchestration.
- Weight sharing works in OpenVINO IR.
- Intel Arc 140V runs the fused DiT action head at about 54.39 ms per 4-step action chunk for G1-shaped inputs.

What is not valid to claim yet:

- Real robot policy accuracy.
- Full VLA latency.
- Qwen2.5-VL latency.
- Trained checkpoint parity.

Planned fix:

- Identify official pretrained UnifoLM-VLA checkpoints.
- Confirm which weights are loaded in each benchmark.
- Label results as either:
  - structural synthetic benchmark,
  - trained checkpoint benchmark,
  - full VLA benchmark.

## What VTune Profiling Is

VTune is Intel's performance profiler. It shows where time is spent when code runs on Intel CPUs and GPUs.

For this project, VTune answers:

```text
The fused DiT graph takes ~54 ms on Intel Arc 140V. What actually happens during those 54 ms?
```

It can help identify:

- top GPU kernels,
- kernel durations,
- EU/GPU utilization,
- memory bandwidth pressure,
- host/device synchronization,
- whether many small kernels are being dispatched,
- whether matmul/attention dominates,
- whether normalization/AdaLayerNorm/MVN is a real hotspot.

Why this matters:

- The proposal says Weeks 5-6 are for VTune profiling.
- Mentor explicitly asked us to start setting up profiling.
- We should not start kernel contribution work until profiling confirms the target.
- AdaLayerNorm is a proposal target, but profiling must prove it is worth optimizing.

## Mentor Feedback and Action Items

### Confirmed by Mentor

- Native source patching direction is good.
- MAE miss is not a serious blocker at this stage.
- Fused-loop experiment is valuable.
- Weight sharing result is important.
- Intel hardware benchmark should be rerun and sent by email.
- VTune/profiling setup should begin.

### Additional Mentor Requests

1. Run the same benchmark on Intel hardware.
   - Status: done for DiT action head.

2. Compare single-step IR and fused-loop IR.
   - Need op counts, graph differences, and fusion/layout analysis.

3. Study OpenVINO optimization strategy.
   - Understand what changed in graph structure after fusing.

4. Analyze FLOPs more rigorously.
   - Do not only rely on AI-generated numbers.
   - Review per-layer formulas.
   - Repeat roofline on Intel GPU.
   - Also analyze the VLM.

5. Investigate VLM-to-DiT boundary.
   - Try combining into one graph if possible.
   - Or use zero-copy tensor handoff to avoid host/device copies.

6. Start VTune profiling.
   - Use results to decide first kernel contribution.

7. Move quickly because hardware reservation may expire.
   - Save results and request another reservation early.

## Unclear Items From Transcript

These should be clarified later:

- Exact OpenVINO zero-copy API the mentor had in mind.
- Exact PyTorch FLOPs library the mentor mentioned.
- Whether mentors expect a full VLM export before kernel work or whether DiT profiling is enough to proceed.
- Whether current benchmark should be rerun with real pretrained action-head weights before presentation.
- Team-wide presentation date, format, length, and audience expectations.
- GSoC admin/onboarding/WordPress requirements.

## Prioritized Checklist

### A. Documentation and Narrative

- [x] Document Intel hardware baseline.
- [x] Document deep benchmark results.
- [x] Document VTune availability blocker and OpenVINO profiling fallback.
- [x] Document presentation-safe claims and current limitations.
- [ ] Keep this project log updated after every experiment.
- [ ] Add a decision log for major architecture choices.
- [x] Convert notes into mentor presentation material.

### B. Mentor Reporting

- [x] Prepare benchmark summary for mentors.
- [x] Prepare update presentation draft.
- [ ] Send mentors both benchmark results and presentation.
- [ ] Include DiT benchmark numbers, VLM blocker, NPU status, and next profiling plan.

### C. Presentation

- [x] Draft full progress presentation narrative.
- [x] Include proposal goals, blockers, native patching, parity, fused-loop, weight sharing, Intel benchmark, VLM blocker, and VTune plan.
- [ ] Prepare a shorter version for broader team presentation.
- [ ] Ask mentors/admin when the team-wide presentation is happening and what format is expected.

### D. VTune Learning and Setup

- [x] Learn VTune basics.
- [x] Write a short note explaining VTune profiling.
- [x] Verify VTune availability on the Intel machine.
- [ ] Run VTune smoke test.
- [ ] Profile fused-loop DiT GPU path.
- [ ] Profile Python-loop DiT GPU path.
- [ ] Compare traces.
- [x] Document that VTune CLI is missing on the runner `PATH`.
- [x] Add OpenVINO node profiling fallback.
- [x] Document OpenVINO node profiling results.

### E. IR Graph Analysis

- [x] Write script to compare single-step IR vs fused-loop IR.
- [x] Count total nodes and op types.
- [x] Identify attention representation.
- [x] Identify AdaLayerNorm/MVN pattern.
- [x] Compare graph differences and possible fusion changes.
- [x] Document what OpenVINO optimization strategy appears to do.

### F. FLOPs and Roofline

- [x] Ask user to re-explain the mentor FLOPs discussion before starting detailed work.
- [x] Document the learning targets for FLOPs, `T_math`, `T_mem`, compute-bound, and communication-bound analysis.
- [ ] Learn formulas for Linear/MatMul, attention, MLP, LayerNorm/MVN, and elementwise ops in enough detail to explain them live.
- [ ] Validate DiT FLOPs manually or with a trusted profiler.
- [ ] Repeat roofline for Intel Arc 140V.
- [ ] Analyze VLM FLOPs/roofline separately.
- [ ] Document assumptions and uncertainty.

### G. Pretrained Weights

- [ ] Identify whether current action-head export loads real pretrained weights.
- [ ] Identify official UnifoLM-VLA checkpoints.
- [ ] Load real action-head weights if available.
- [ ] Rerun parity and benchmark with real weights.
- [ ] Clearly label structural vs trained-checkpoint benchmarks.

### H. VLM / Full VLA Benchmarking

- [ ] Patch `convert_qwen_vlm.py` to disable mock fallback by default.
- [ ] Identify exact Qwen2.5-VL checkpoint.
- [ ] Install missing dependencies.
- [ ] Try real VLM export.
- [ ] Benchmark VLM alone once real IR exists.
- [ ] Benchmark full VLA once VLM and DiT can run together.

### I. VLM-to-DiT Boundary / Zero Copy

- [ ] Investigate OpenVINO zero-copy/shared tensor APIs.
- [ ] Measure current tensor handoff overhead.
- [ ] Prototype zero-copy VLM output to DiT input.
- [ ] Compare separate models vs zero-copy vs combined graph.

### J. NPU

- [x] Document current NPU dynamic-shape compiler failure.
- [ ] Defer NPU until CPU/GPU path is stable.
- [ ] Later test static-shape export for NPU.

### K. Admin

- [ ] Follow up on GSoC/OpenVINO onboarding material.
- [ ] Ask about WordPress/blog requirements.
- [ ] Track hardware reservation expiration.
- [ ] Request extension/new reservation early.

## Recommended Next Work Order

Tonight, prioritize:

1. Mentor update email and presentation outline.
2. VTune learning note.
3. IR graph comparison script.
4. VLM export cleanup.
5. FLOPs/roofline learning note.
6. VTune setup/run if hardware and tools are available.

Do not get stuck on full VLM export tonight if weights/access are not immediately available. Record the blocker and move on.
