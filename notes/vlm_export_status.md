# VLM Export Status

## Why This Matters

The DiT action-head benchmark is now valid on Intel CPU/GPU, but full VLA latency requires the VLM stage:

```text
image + text + robot state
-> Qwen2.5-VL backbone
-> VLM hidden states
-> fused DiT action head
-> action chunk
```

So far, our valid benchmark starts after the VLM:

```text
synthetic VLM hidden states
-> fused DiT action head
-> action chunk
```

This means current Intel latency numbers are DiT action-head numbers, not full VLA numbers.

## Previous State

`export_tests/convert_qwen_vlm.py` originally attempted to load the real Qwen2.5-VL model and silently fell back to a dummy mock model if loading failed.

That was useful for checking whether the conversion path had the right general shape, but it created a risk:

```text
mock VLM artifact could be mistaken for real Qwen2.5-VL export
```

The local/mock `qwen_vlm_backbone.bin` was only bytes-scale, not GB-scale. A real Qwen2.5-VL artifact should be large.

## Decision

Mock fallback is now disabled by default.

Current command:

```bash
python export_tests/convert_qwen_vlm.py
```

Expected behavior:

- loads real Qwen2.5-VL interface,
- converts real model,
- writes real OpenVINO IR,
- fails if real model loading fails.

Structural mock mode is still available, but must be explicit:

```bash
python export_tests/convert_qwen_vlm.py --allow-mock
```

Mock mode must not be used for:

- VLM latency,
- full VLA latency,
- trained-model accuracy,
- presentation claims about end-to-end performance.

## What We Need Next

1. Identify the exact Qwen2.5-VL checkpoint used by UnifoLM-VLA.
2. Verify whether weights are public, gated, or stored elsewhere.
3. Install any missing dependencies on the Intel hardware.
4. Run real VLM export.
5. Confirm the `.bin` file is GB-scale.
6. Benchmark VLM alone.
7. Benchmark full VLA as:

```text
VLM latency + VLM-to-DiT handoff + fused DiT latency
```

## Presentation Language

Use this wording:

```text
We have validated and benchmarked the DiT action-head path on Intel GPU. Full VLA latency is not yet reported because the real Qwen2.5-VL backbone export is still pending. The previous VLM artifact was only a structural mock and is now explicitly guarded against accidental benchmark use.
```
