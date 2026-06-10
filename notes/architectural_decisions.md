# Architectural Decision: Single-Step vs. Full-Loop Compilation

## The Question
Why compile a single-step DiT rather than tracing the entire DiT component with all four denoising steps included in the graph?

## The Reasoning

When we use standard PyTorch tracing (`torch.jit.trace` or OpenVINO's equivalent converter) on a model that contains a Python `for` loop, the tracer does not understand the concept of a loop. Instead, it **"unrolls"** the loop.

If we traced the full 4-step DiT loop, we would not get a graph with a loop. We would get a graph with **four sequential copies** of the entire DiT architecture hardcoded into it.

### 1. Memory and Graph Size
- **Single-Step IR**: The DiT model has ~550 million parameters. The exported IR (the `.bin` file) is roughly **1.0 GB** in FP16.
- **Full-Loop IR**: While tracing an unrolled 4-step loop creates a graph with four sequential copies of the transformer blocks, OpenVINO successfully **shares the weights** across these steps. Consequently, the `fused_loop_dit.bin` remains **1.0 GB**, same as the single-step version. The primary bloat is in the `.xml` graph description (2.2MB vs 0.7MB), which is negligible.

### 2. Performance (Fusion Speedup)
- **Python-Orchestrated Loop**: 495.65 ms
- **OpenVINO Fused Loop**: 259.50 ms
- **Observation**: The Fused Loop is nearly **2x faster**. This is likely due to the elimination of Python-to-OpenVINO call overhead and potential cross-step optimizations (e.g., operator fusion, improved scheduling) that the OpenVINO compiler can perform on the larger static graph.

### 3. Compilation Latency
- The OpenVINO compiler (and downstream hardware plugins like oneDNN or the GPU compiler) needs to optimize the graph.
- A graph that is 4x larger takes longer to optimize and compile at runtime, though for a 4-step unroll, this remains within acceptable limits for most deployment scenarios.

### 4. Loss of Flexibility (Dynamic Horizon)
- In flow-matching and diffusion models, the number of inference timesteps ($N$) is a tunable hyperparameter. 
- You might want 4 steps for fast execution or 8 steps for higher precision trajectory refinement.
- **If we trace the loop**: The graph is permanently locked to $N=4$. To change it to 8 steps, we would have to re-export and recompile an 8-step graph.
- **With a single-step wrapper**: The OpenVINO IR represents just the math of one step. The Python scheduler dictates how many times to call it. We gain complete flexibility over the action horizon without touching the compiled model.

## Summary for Mentors
"While isolating a single step provides maximum flexibility for research and dynamic action horizons, our benchmarks show that a **Fused Loop** (static unrolling) delivers a **91% speedup** (259ms vs 495ms) with **zero additional weight memory overhead** (both are ~1.0GB). We have decided to support both: a single-step IR for flexible refinement and a fused-loop IR for high-performance production inference."
