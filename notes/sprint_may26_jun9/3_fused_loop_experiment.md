# Fused Loop Experiment (Compiling Orchestration)

## Goal
Empirically test the mentors' hypothesis: does tracing the entire 4-step denoising loop into a single OpenVINO IR allow for compiler fusions that beat the "Single-Step + Python Loop" architecture?

## Theoretical Rationale
- **The "Fused" Argument**: By presenting the OpenVINO graph optimizer with all 4 steps at once, it can perform aggressive inter-block fusions, reorder memory layouts, and potentially reduce the kernel dispatch count. This eliminates the "bubbles" (Python-to-C++ transition) between steps.
- **The "Single-Step" Argument**: Tracing a loop unrolls it, duplicating the 550M parameters 4 times. This causes massive memory bloat (4.4GB vs 1.1GB) which might saturate memory bandwidth and slow down compilation/loading.

## Mentor Alignment
- **Transcript**: Mentors suggested "fusing the component as much as possible" and warned about "bubbles" between steps.
- **Target**: Benchmark and compare the two strategies on CPU.

## Execution Details
- **Wrapper**: `export_tests/FullLoopDiTWrapper.py` (Unrolls the `for` loop).
- **Export Script**: `export_tests/convert_fused_loop_dit.py`.
- **Metrics**: IR file size (`.bin`), compilation time, and steady-state latency.

## Failure Prediction
1. **Memory Overflow**: The conversion might crash on machines with < 16GB RAM during the unrolling phase.
2. **Compile Time Blowup**: `core.compile_model()` might take minutes instead of seconds.
3. **Diminishing Returns**: If the DiT is already compute-bound (as hypothesized), reducing orchestration bubbles will have negligible impact on total latency.

## Results Achieved
- **Python-Orchestrated Loop**: 495.65 ms
- **OpenVINO Fused Loop**: **259.50 ms**
- **Speedup**: **91.0%**
- **IR Weight Size**: 1.1 GB (Constant across both approaches)
- **IR Graph Size**: 2.2 MB (XML tripled in complexity)

## Conclusion
The results provide overwhelming evidence in favor of the **Fused Loop** architecture. By unrolling the 4 steps into a single static graph, we eliminated approximately 236ms of overhead per action chunk. 

Crucially, the experiment disproved the "Memory Bloat" fear: OpenVINO successfully implemented **Weight Sharing**, ensuring that the unrolled graph pointed to the same 550M parameter buffer. This achieves the mentors' goal of "fusing compute" without the 4.4GB memory penalty.

## Presentation Highlights
- **Show the 91% speedup metric.**
- **Explain the "Weight Sharing" insight**: This is a key systems engineering detail that justifies why the "Fused" approach is safe for deployment.
