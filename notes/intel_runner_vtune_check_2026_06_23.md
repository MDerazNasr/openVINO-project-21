# Intel Runner VTune Availability Check - June 23, 2026

## Context

After adding the VTune profiling harness, the `Intel Hardware Benchmark` workflow ran automatically from commit:

```text
c6d184ad add VTune profiling harness
```

Workflow run:

```text
https://github.com/MDerazNasr/openVINO-project-21/actions/runs/27985830438
```

Artifact:

```text
https://github.com/MDerazNasr/openVINO-project-21/actions/runs/27985830438/artifacts/7805702494
```

## Result

The benchmark succeeded.

The VTune availability check also ran, but VTune was not available on the runner `PATH`:

```text
VTune CLI was not found on PATH.
Set run_vtune=false for benchmark-only runs, or install Intel VTune Profiler on the runner.
```

The opt-in `Run VTune GPU Hotspots` step was skipped because this was a push-triggered benchmark run, not a manual workflow dispatch with `run_vtune=true`.

## Latest Benchmark Numbers

OpenVINO:

```text
2026.2.1-21919-ede283a88e3-releases/2026/2
```

Devices:

```text
CPU, GPU, NPU
```

Hardware:

```text
CPU: Intel Core Ultra 7 258V
GPU: Intel Arc 140V GPU (16GB) iGPU
NPU: Intel AI Boost
```

Deep benchmark summary:

| Device | Single Step Mean | Python Loop Mean | Fused Loop Mean | Fused Speedup | Fused Chunks/s | Status |
|---|---:|---:|---:|---:|---:|---|
| CPU | 305.92 ms | 1254.21 ms | 871.28 ms | 1.44x | 1.15 | ok |
| GPU | 15.90 ms | 63.68 ms | 55.26 ms | 1.15x | 18.10 | ok |
| NPU | n/a | n/a | n/a | n/a | n/a | skipped |

Full VLA / VLM status from the run:

```text
No Qwen2.5-VL .bin artifact is present; VLM and full VLA latency are blocked.
```

## Interpretation

The profiling harness itself is integrated and the normal benchmark path still passes.

The next blocker is tooling availability:

```text
VTune is not currently callable as `vtune` or `amplxe-cl` from the runner environment.
```

This means we cannot collect VTune GPU Hotspots from GitHub Actions yet.

## Decision

Do not run the manual `run_vtune=true` workflow until one of these is true:

1. VTune is installed and added to `PATH` on the Intel runner.
2. We add a workflow step that locates VTune in a known install directory.
3. We decide to install VTune during the workflow, if allowed and practical.

Until then, use the existing OpenVINO benchmark and IR graph analysis to continue documentation, but do not claim VTune profiling results.

## Next Options

### Option A: Install or Locate VTune

Actions:

- Check whether Intel VTune Profiler is installed somewhere outside `PATH`.
- If installed, add its CLI path to the workflow.
- If not installed, determine whether we can install it on the reserved machine.

This is the preferred route if we want true VTune GPU Hotspots.

### Option B: Use OpenVINO Runtime Profiling First

Actions:

- Enable OpenVINO per-layer/per-node profiling on the compiled model.
- Collect profiling info for fused-loop and Python-loop modes.
- Use this as an interim hotspot map while VTune is unavailable.

This will not replace VTune, but it can still identify expensive OpenVINO graph nodes and help decide whether MVN/AdaLayerNorm appears significant.

### Option C: Ask Mentor About Tooling

Actions:

- Report that the self-hosted Intel runner does not have VTune on `PATH`.
- Ask whether there is an expected VTune install path, module setup command, or alternative profiling workflow.

This is useful because the hardware reservation environment may have a standard Intel tooling setup that is not automatically loaded in GitHub Actions.
