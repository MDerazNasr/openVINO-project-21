# Roofline Performance Analysis

## Goal
Establish the theoretical hardware limits of the current development machine (Apple M4) and the target Intel platform to determine the optimization "headroom" for the UnifoLM-VLA model.

## Theoretical Rationale
The **Roofline Model** is a performance model used to provide performance estimates of a given compute kernel or application. It relates processor performance to the **Arithmetic Intensity** (AI), which is the ratio of floating-point operations (FLOPs) to memory traffic (Bytes).

By calculating the AI of our DiT Action Head, we can determine whether our optimization efforts should focus on **Compute** (improving kernel efficiency) or **Memory Bandwidth** (reducing weight movement).

## Hardware Specifications (Baselines)

| Hardware | Peak Performance (FP32) | Memory Bandwidth |
|---|---|---|
| **Apple M4 (CPU)** | ~1.5 - 2.0 TFLOPS | 120 GB/s |
| **Intel Arrow Lake (iGPU)** | *Pending Cloud Access* | *TBD* |

## Model Arithmetic Intensity (DiT Head)

### 1. Operation Count (FLOPs)
- **Parameters**: 550,386,688
- **Joint Sequence Length (LIBERO)**: 41 tokens
- **FLOPs Estimation**: $2 \times \text{Params} \times \text{SeqLen}$
- **Total FLOPs per step**: $\approx 45.1 \text{ GFLOPs}$

### 2. Memory Traffic (Bytes)
- **Model Size (FP32)**: 1.1 GB
- **Bytes Transferred per step**: $\approx 1.1 \text{ GB}$

### 3. Arithmetic Intensity (AI)
- $AI = \frac{\text{GFLOPs}}{\text{GB Transferred}} = \frac{45.1}{1.1} \approx \mathbf{41 \text{ FLOPs/Byte}}$

## Results & "Headroom" Analysis

- **Actual Throughput**: Based on 106ms latency, our current throughput is **~0.425 TFLOPS**.
- **Performance vs. Roofline**: We are operating at roughly **21% - 28%** of the Apple M4 CPU's peak capacity.
- **Limiting Factor**: With an AI of 41, the model is heavily **Compute-Bound**.

## Mentor Alignment
- **Transcript**: Mentors specifically requested checking the "Roofline performance" and identifying the "upper limit."
- **Observation**: Our current position confirms the mentors' hypothesis that there is significant room (70%+) for optimization via kernel fusions (`AdaLayerNorm`) and parallel execution.

## Presentation Highlights
- Present the AI of 41 to prove the model is Compute-Bound.
- Show that we are currently at ~25% of the hardware's theoretical limit, justifying the optimization phase.
