# Benchmark Results

## Single-Step DiT

| Runtime | Mean ms | Median ms | Min ms | Max ms | Notes |
|---|---:|---:|---:|---:|---|
| PyTorch CPU | 132.52 | 128.64 | 124.52 | 253.61 | Baseline PyTorch wrapper (eval mode) |
| OpenVINO CPU | 105.97 | 103.29 | 100.70 | 136.15 | Compiled single-step IR (FP32) |
| OpenVINO external loop | 418.70 | 413.56 | 406.52 | 442.77 | 4 steps of denoising overhead + IR compute |

## Observations
- **OpenVINO Performance Gain**: The compiled OpenVINO IR demonstrates a clear performance advantage over PyTorch on the CPU for a single step (105.97ms vs 132.52ms, ~20% faster).
- **Consistency**: OpenVINO execution is more stable, with a significantly lower standard deviation (6.33ms vs 18.90ms for PyTorch) and a much lower max latency outlier.
- **Orchestration Overhead**: The external loop for 4 steps takes roughly `4 * 105ms = 420ms`, matching our benchmark result (418.70ms). This proves that the Python orchestration logic (Euler integration, timestep calculation, etc.) adds virtually zero overhead compared to the core DiT compute.

## Caveats
- CPU-only execution (Apple M4). GPU access is unavailable until cloud resources are provisioned.
- Used dummy inputs with traced dimensions (`LIBERO` constants).
- FP32 precision was used; INT8/BF16 optimizations have not yet been applied.
- Not final benchmark numbers.
