# Architectural Decision: Single-Step vs. Full-Loop Compilation

## The Question
Why compile a single-step DiT rather than tracing the entire DiT component with all four denoising steps included in the graph?

## The Reasoning

When we use standard PyTorch tracing (`torch.jit.trace` or OpenVINO's equivalent converter) on a model that contains a Python `for` loop, the tracer does not understand the concept of a loop. Instead, it **"unrolls"** the loop.

If we traced the full 4-step DiT loop, we would not get a graph with a loop. We would get a graph with **four sequential copies** of the entire DiT architecture hardcoded into it.

### 1. Memory Bloat (Graph Size)
- **Single-Step IR**: The DiT model has ~550 million parameters. The exported IR (the `.bin` file) is roughly **1.1 GB**.
- **Full-Loop IR**: An unrolled 4-step graph would duplicate those operations. The resulting memory footprint would skyrocket, potentially exceeding **4 GB**. This is highly inefficient and creates immense pressure on the deployment hardware's memory bandwidth.

### 2. Compilation Latency
- The OpenVINO compiler (and downstream hardware plugins like oneDNN or the GPU compiler) needs to optimize the graph.
- A graph that is 4x larger takes exponentially longer to optimize and compile at runtime.

### 3. Loss of Flexibility (Dynamic Horizon)
- In flow-matching and diffusion models, the number of inference timesteps ($N$) is a tunable hyperparameter. 
- You might want 4 steps for fast execution or 8 steps for higher precision trajectory refinement.
- **If we trace the loop**: The graph is permanently locked to $N=4$. To change it to 8 steps, we would have to re-export and recompile an 8-step graph.
- **With a single-step wrapper**: The OpenVINO IR represents just the math of one step. The Python scheduler dictates how many times to call it. We gain complete flexibility over the action horizon without touching the compiled model.

### 4. Focused Optimization Target
- By isolating the single-step compute (which takes ~105ms), we isolate the true bottleneck.
- Our efforts (like fusing `AdaLayerNorm` or optimizing attention) can be focused on this single, reusable block of compute. Any improvement here is automatically multiplied by $N$ during inference.

## Summary for Mentors
"Tracing a Python loop usually results in static unrolling, which would duplicate the 550M parameter DiT four times in the OpenVINO IR, causing massive memory bloat and long compile times. By isolating a single step, we keep the model footprint small (1.1GB), ensure the optimization target is tightly scoped, and retain the flexibility to change the number of denoising steps dynamically at runtime."
