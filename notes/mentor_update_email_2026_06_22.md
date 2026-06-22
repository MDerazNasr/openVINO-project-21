# Mentor Update Email Draft - June 22, 2026

Subject: OpenVINO VLA Intel Hardware Benchmark Results and Next Profiling Plan

Hi [Mentor Names],

I wanted to send a concise update after getting access to the Intel hardware and rerunning the OpenVINO benchmarks on the target system.

## Hardware / Runtime

The benchmark ran on:

- CPU: Intel Core Ultra 7 258V
- GPU: Intel Arc 140V GPU (16GB) iGPU
- NPU: Intel AI Boost
- OpenVINO: 2026.2.1
- Available OpenVINO devices: CPU, GPU, NPU

## What Was Benchmarked

This benchmark covers the DiT action-head path:

```text
VLM hidden-state tensor / vl_embs
-> DiT action head
-> 4-step denoising
-> action chunk
```

It does not yet include the full Qwen2.5-VL backbone. The current VLM IR artifact in the repo is only a mock/template export, so I am not treating it as a valid full VLA latency number.

## DiT Benchmark Results

The benchmark used the G1 configuration:

- NUM_ACTIONS_CHUNK = 25
- ACTION_DIM = 23
- PROPRIO_DIM = 23

### Intel CPU

- Single DiT step mean: 307.54 ms
- Python-orchestrated 4-step loop mean: 1223.56 ms
- Fused 4-step OpenVINO IR mean: 873.17 ms
- Fused speedup vs Python loop: 1.40x
- Fused throughput: 1.15 action chunks/sec

### Intel Arc 140V GPU

- Single DiT step mean: 16.20 ms
- Python-orchestrated 4-step loop mean: 65.31 ms
- Fused 4-step OpenVINO IR mean: 54.39 ms
- Fused speedup vs Python loop: 1.20x
- Fused throughput: 18.39 action chunks/sec
- Fused p95 latency: 55.58 ms
- Fused p99 latency: 55.93 ms

### NPU

I skipped NPU in the final benchmark run because the dynamic DiT graph previously caused the NPU compiler to abort during shape/type inference. I am treating NPU as a separate static-shape enablement task after the CPU/GPU path is stable.

## Main Takeaways

1. The Intel Arc 140V iGPU is currently the viable target for the DiT action-head path.
2. The fused-loop IR is still faster than Python orchestration on target hardware.
3. OpenVINO weight sharing still holds on the Intel run: the fused-loop `.bin` is effectively the same size as the single-step `.bin`, while the XML graph grows.
4. The relative fused-loop speedup on GPU is smaller than the earlier CPU-only experiment, but the absolute GPU latency is much better.
5. Full VLA latency is still blocked on exporting/benchmarking the real Qwen2.5-VL backbone.

## Next Plan

Based on the proposal timeline and our last meeting, I think the next step is to move into profiling:

1. Set up VTune/OpenVINO profiling for the fused DiT GPU path.
2. Compare fused-loop vs Python-loop traces.
3. Identify whether the main hotspot is attention/matmul, AdaLayerNorm/MVN decomposition, memory movement, or dispatch/synchronization overhead.
4. Use that profiling result to decide the first kernel/runtime contribution target.

In parallel, I will work on the full VLA path:

1. Make the Qwen2.5-VL export script fail loudly instead of silently falling back to a mock model.
2. Identify/load the real Qwen2.5-VL checkpoint.
3. Export and benchmark the VLM backbone separately.
4. Then measure VLM + fused DiT end-to-end latency.
5. Investigate zero-copy or shared-tensor handoff between VLM output and DiT input, or a combined graph if feasible.

I also plan to analyze the single-step IR vs fused-loop IR more deeply by comparing op counts, graph structure, attention representation, and AdaLayerNorm/MVN decomposition. This should help me better understand OpenVINO's graph optimization strategy before starting kernel work.

Best,
Mohamed

## Notes for Personal Follow-Up

- Attach or link `notes/intel_hardware_baseline_2026_06_22.md`.
- Attach or link GitHub Actions benchmark artifact if useful.
- Add slides once presentation draft exists.
- Ask whether mentors want a short interim meeting before the next scheduled meeting.
- Ask whether they want the next update to prioritize VTune trace results or full VLM export first.
