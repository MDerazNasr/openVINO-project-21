# Mentor Update Email Draft - June 23, 2026

## Subject

```text
OpenVINO VLA update: Intel DiT benchmark, profiling fallback, and next steps
```

## Email Draft

```text
Hi [Mentor Name],

I wanted to send a concise update on the hardware benchmark and profiling work.

I now have the DiT action-head benchmark running reproducibly on the Intel machine through a GitHub Actions self-hosted runner. The target system reports:

- CPU: Intel Core Ultra 7 258V
- GPU: Intel Arc 140V GPU (16GB) iGPU
- NPU: Intel AI Boost
- OpenVINO: 2026.2.1

Latest DiT action-head results:

| Device | Single Step Mean | Python 4-Step Loop | Fused 4-Step IR | Fused Speedup | Fused Chunks/s |
|---|---:|---:|---:|---:|---:|
| CPU | 306.02 ms | 1253.17 ms | 869.47 ms | 1.44x | 1.15 |
| GPU | 15.93 ms | 63.88 ms | 54.48 ms | 1.17x | 18.36 |
| NPU | n/a | n/a | n/a | n/a | skipped |

The fused-loop result is still important because the 4-step graph keeps approximately the same `.bin` weight size as the single-step graph. The XML graph grows, but the weight file stays around 1.12 GB, so OpenVINO appears to share/deduplicate the repeated weights instead of storing four copies.

I also added a VTune profiling harness and an opt-in workflow path, but VTune CLI is not currently available on the runner PATH. To keep moving, I added OpenVINO node profiling with `PERF_COUNT=YES` as an interim hotspot view.

The first OpenVINO GPU profile suggests the fused DiT graph is dominated by:

- MLP FullyConnected: 52.15%
- self-attention projections: 26.01%
- other FullyConnected: 11.63%
- SDPA attention: 4.42%
- MVN / normalization: 2.21%

So MVN/AdaLayerNorm is measurable, but the current profile does not show it as the main bottleneck. I think the safe next step is to either get VTune working or deepen the OpenVINO profile before choosing the first kernel contribution target.

One clarification: the current benchmark is DiT-only with synthetic VLM embeddings. I am not claiming full end-to-end VLA latency yet because the repo does not currently have a real Qwen2.5-VL OpenVINO `.bin` artifact. The tiny/template VLM artifact is treated as mock and excluded from full latency reporting.

Questions I would like your guidance on:

1. Should I prioritize getting VTune installed/available on the runner, or continue with deeper OpenVINO profiling first?
2. Given the current profile, should the first kernel target still be AdaLayerNorm/MVN, or should I investigate the FullyConnected / attention projection path first?
3. Which official pretrained UnifoLM-VLA checkpoint should I use for trained-policy benchmarks?
4. What exact input setup should define the full VLA benchmark once Qwen2.5-VL is exported?
5. For the upcoming wider team presentation, what level of FLOPs/roofline detail should I prepare?

I am also preparing a short presentation that summarizes the hardware benchmark, profiling status, full-VLA blocker, and next steps.

Best,
Mohamed
```

## Attachments / Links To Include

- Latest benchmark/profile workflow run:
  `https://github.com/MDerazNasr/openVINO-project-21/actions/runs/27986729224`
- Latest benchmark/profile artifact:
  `https://github.com/MDerazNasr/openVINO-project-21/actions/runs/27986729224/artifacts/7806038666`
- Project log:
  `notes/project_log_and_next_plan.md`
- Profiling note:
  `notes/openvino_node_profile_2026_06_23.md`
- Presentation package:
  `notes/mentor_presentation_package_2026_06_23.md`

## Notes

This email intentionally separates:

- real DiT action-head hardware benchmark,
- interim OpenVINO node profiling,
- missing VTune result,
- blocked full VLA/VLM benchmark,
- unresolved pretrained checkpoint question.

That separation keeps the update technically accurate and avoids overclaiming.
