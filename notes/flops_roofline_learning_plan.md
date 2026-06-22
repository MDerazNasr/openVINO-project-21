# FLOPs, Arithmetic Intensity, and Roofline Learning Plan

## Why This Note Exists

In the mentor meeting, the feedback was not only to calculate a roofline number. The goal is to understand the theory:

- what FLOPs are,
- how FLOPs are calculated,
- how model layers translate into computation,
- what `T_math` and `T_comm` / `T_mem` mean,
- what compute-bound and communication-bound mean,
- how arithmetic intensity connects computation to memory traffic,
- how all of this applies to OpenVINO and this VLA optimization project.

This note is a learning plan and a technical checklist for validating the roofline analysis instead of treating it as an AI-generated number.

## The Core Idea

Every model inference does two broad things:

1. Performs math.
2. Moves data.

The time for a layer or model is often limited by whichever is slower:

```text
T_total ~= max(T_math, T_mem)
```

Where:

```text
T_math = FLOPs / hardware_compute_throughput
T_mem  = bytes_moved / memory_bandwidth
```

Some people call the memory/data movement term `T_comm`, `T_mem`, or communication time. In this project, the meaning is:

```text
time spent moving tensors/weights/activations rather than doing math
```

## What FLOPs Are

FLOPs means floating-point operations.

Examples:

```text
a + b     -> 1 FLOP
a * b     -> 1 FLOP
a * b + c -> often counted as 2 FLOPs
```

For neural networks, FLOPs estimate how much numerical work is required by a layer or model.

Important:

- FLOPs are not latency.
- FLOPs are a hardware-independent workload estimate.
- Latency depends on hardware, compiler, memory, layout, precision, and kernels.

## Why FLOPs Matter For This Project

The project is about making VLA inference fast on Intel hardware.

FLOPs help answer:

```text
How much math does the DiT action head require?
How much math does the VLM require?
Is the workload big enough to use the GPU well?
Is the bottleneck likely math or data movement?
How much theoretical speedup is possible?
```

But FLOPs alone are not enough. A model can have many FLOPs but still be slow because memory movement dominates.

That is why we also need arithmetic intensity and roofline analysis.

## Arithmetic Intensity

Arithmetic intensity is:

```text
Arithmetic Intensity = FLOPs / bytes moved
```

Units:

```text
FLOPs per byte
```

Interpretation:

- High arithmetic intensity means lots of math per byte loaded.
- Low arithmetic intensity means lots of data movement per unit of math.

Transformer matmuls often have relatively high arithmetic intensity. Elementwise ops and normalization often have lower arithmetic intensity because they touch memory but do little math.

## Compute-Bound vs Communication-Bound

### Compute-Bound

A workload is compute-bound when:

```text
T_math > T_mem
```

Meaning:

- math units are the bottleneck,
- making memory faster may not help much,
- improving kernels, precision, matmul efficiency, or fusion may help.

### Communication-Bound / Memory-Bound

A workload is communication-bound or memory-bound when:

```text
T_mem > T_math
```

Meaning:

- data movement is the bottleneck,
- reducing memory traffic matters,
- fusion can help by avoiding intermediate writes/reads,
- zero-copy handoff can help if tensors otherwise move through host memory.

In the meeting, the mentor wanted this distinction understood because it determines what kind of optimization is worth doing.

## Roofline Model

The roofline model relates arithmetic intensity to achievable performance.

It uses two hardware limits:

```text
peak compute throughput
peak memory bandwidth
```

The roofline says:

```text
achievable performance <= min(peak_compute, arithmetic_intensity * memory_bandwidth)
```

This creates two regions:

1. Memory-bound region:
   - arithmetic intensity is low,
   - performance increases if arithmetic intensity improves.

2. Compute-bound region:
   - arithmetic intensity is high,
   - performance is limited by peak compute.

The ridge point is:

```text
ridge_point = peak_compute / memory_bandwidth
```

If:

```text
arithmetic_intensity > ridge_point
```

then the workload is compute-bound.

If:

```text
arithmetic_intensity < ridge_point
```

then the workload is memory/communication-bound.

## How This Connects To OpenVINO

OpenVINO changes how the model runs:

```text
PyTorch model
-> OpenVINO IR
-> graph optimizations
-> device plugin compile
-> hardware kernels
```

OpenVINO can affect both sides of the roofline:

### It can reduce memory movement

Examples:

- operator fusion,
- layout optimization,
- avoiding unnecessary transposes/reorders,
- constant folding,
- zero-copy tensor handoff.

This improves the `T_mem` / `T_comm` side.

### It can improve math efficiency

Examples:

- optimized matmul kernels,
- fused attention kernels,
- lower precision such as FP16/BF16/INT8,
- better kernel scheduling,
- hardware-specific plugin optimizations.

This improves the `T_math` side.

### It can reduce orchestration overhead

The fused-loop experiment is an example:

```text
Python loop calling OpenVINO 4 times
```

vs:

```text
one fused OpenVINO graph containing 4 denoising steps
```

This is not only a FLOPs issue. It affects dispatch overhead, scheduling, and intermediate tensor lifetime.

## How This Applies To The DiT Action Head

The DiT action head contains:

- action/state encoders,
- transformer blocks,
- self-attention,
- cross-attention against VLM embeddings,
- feed-forward/MLP layers,
- AdaLayerNorm/MVN + timestep-conditioned scale/shift,
- action decoder,
- Euler-style denoising update.

Expected high-FLOP parts:

- MatMul / Linear layers,
- attention projections,
- attention score/value products,
- feed-forward MLPs.

Expected lower-FLOP but memory-sensitive parts:

- MVN / LayerNorm,
- Add,
- Multiply,
- Reshape,
- Transpose,
- Convert,
- Concats/splits.

This matters because our IR comparison showed:

```text
ScaledDotProductAttention: 16 -> 64
MVN:                       33 -> 132
MatMul:                    123 -> 438
Add:                       191 -> 711
Convert:                   272 -> 676
```

So the fused graph structurally repeats both heavy math ops and normalization/elementwise ops.

VTune should decide which of these dominate runtime.

## How This Applies To The VLM

The VLM is likely much larger than the DiT action head.

Expected VLM characteristics:

- many transformer layers,
- large hidden size,
- long sequence length,
- vision encoder patches,
- attention and MLP-heavy compute,
- large weight memory footprint.

Full VLA latency may be dominated by:

```text
Qwen2.5-VL feature extraction
```

or:

```text
repeated DiT denoising
```

We cannot know until the real VLM is exported and benchmarked.

That is why the mentor asked whether we analyzed the VLM too. We have not done that yet, and it should be added.

## Layer FLOPs Formulas To Learn

### Linear / Fully Connected

For:

```text
input:  [B, T, K]
weight: [K, N]
output: [B, T, N]
```

Approximate FLOPs:

```text
2 * B * T * K * N
```

The factor of 2 counts multiply and add.

### MatMul

For:

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

Approximate FLOPs:

```text
2 * M * K * N
```

### Attention

Attention has multiple parts:

1. Q/K/V projections.
2. Attention scores:

```text
Q @ K^T
```

3. Softmax.
4. Weighted values:

```text
softmax(QK^T) @ V
```

5. Output projection.

The exact FLOPs depend on:

- batch size,
- sequence length,
- number of heads,
- head dimension,
- whether attention is self-attention or cross-attention.

### MLP / Feed-Forward

Transformer MLPs are often:

```text
hidden -> expansion -> hidden
```

Approximate FLOPs:

```text
2 * B * T * hidden * expansion
+ 2 * B * T * expansion * hidden
```

Activation functions add smaller extra cost.

### LayerNorm / MVN

LayerNorm/MVN usually includes:

- mean,
- variance,
- subtract,
- divide/sqrt,
- scale,
- shift.

FLOPs are much smaller than large matmuls, but memory traffic can be significant because the tensor is read/written multiple times if not fused.

This is why AdaLayerNorm can be important even if it is not FLOP-heavy.

### Elementwise Ops

Examples:

```text
Add
Multiply
Convert
GELU/SiLU
```

These are often low arithmetic intensity:

```text
few FLOPs per byte moved
```

Fusion can help because it avoids writing intermediate tensors to memory.

## What We Need To Validate

The earlier roofline analysis claimed:

```text
DiT arithmetic intensity ~= 41 FLOPs/byte
compute-bound
~75% theoretical headroom
```

We need to validate:

1. Which dimensions were used?
2. Which layers were included?
3. Were multiply-adds counted as 1 FLOP or 2 FLOPs?
4. Were attention softmax and normalization included?
5. Was memory traffic estimated from weights only or weights + activations?
6. Was the hardware peak based on Apple M4, Intel CPU, or Intel GPU?
7. Is the same conclusion true on Intel Arc 140V?

## Validation Checklist

- [ ] Recompute DiT FLOPs from tensor shapes.
- [ ] Separate FLOPs by module:
  - action encoder,
  - state encoder,
  - self-attention,
  - cross-attention,
  - MLP,
  - AdaLayerNorm/MVN,
  - action decoder.
- [ ] Estimate bytes moved:
  - weights,
  - activations,
  - intermediate tensors,
  - repeated denoising steps.
- [ ] Compute arithmetic intensity.
- [ ] Get Intel Arc 140V theoretical peak compute.
- [ ] Get Intel Arc 140V memory bandwidth estimate.
- [ ] Compute ridge point.
- [ ] Determine if DiT is compute-bound or memory-bound on Intel GPU.
- [ ] Compare predicted roofline performance with measured:

```text
GPU fused DiT: 54.39 ms
GPU single step: 16.20 ms
```

- [ ] Repeat for VLM once real VLM dimensions/weights are available.

## How This Informs Kernel Work

If DiT is compute-bound:

- focus on math-heavy kernels,
- attention/matmul efficiency,
- precision,
- GPU occupancy,
- SDPA/matmul backend selection.

If DiT is memory-bound:

- focus on fusion,
- reduce intermediate reads/writes,
- eliminate layout conversions,
- zero-copy tensor handoff,
- fuse AdaLayerNorm/MVN + scale + shift.

If AdaLayerNorm is low FLOP but visible in runtime:

- it may be worth fusing because it is memory/dispatch-bound.

If attention dominates:

- AdaLayerNorm fusion may still help, but it should not be the first contribution unless profiling shows it matters.

## Relationship To VTune

Roofline is theory:

```text
Based on FLOPs and bytes, what should limit us?
```

VTune is measurement:

```text
What actually limits us on Intel hardware?
```

The next step is to make them agree or explain why they differ.

Examples:

- Roofline says compute-bound, VTune shows high EU utilization and matmul dominance.
  - Good: theory and measurement agree.

- Roofline says compute-bound, VTune shows low utilization and many small kernels.
  - The model may be dispatch/layout-bound despite high arithmetic intensity.

- Roofline says memory-bound, VTune shows bandwidth saturation.
  - Focus on fusion and memory movement.

## Learning Tasks

- [ ] Review FLOPs formulas for transformer layers.
- [ ] Find and test a PyTorch FLOPs tool if useful.
- [ ] Ask mentors which FLOPs tool/library they had in mind.
- [ ] Compare tool output against manual estimates.
- [ ] Document any mismatch.
- [ ] Create a slide explaining:

```text
FLOPs -> bytes -> arithmetic intensity -> roofline -> optimization target
```

## Presentation Angle

The mentor-facing story should be:

1. We used FLOPs/roofline to reason about theoretical limits.
2. We are validating those numbers layer-by-layer instead of relying blindly on AI output.
3. We will compare roofline theory with VTune measurements.
4. The final optimization target will be chosen from data:
   - AdaLayerNorm/MVN if memory/elementwise patterns matter,
   - attention/matmul if compute dominates,
   - layout/zero-copy if memory movement dominates.
