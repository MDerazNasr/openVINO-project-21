# VTune Profiling Plan

## Why This Note Exists

The proposal's Weeks 5-6 milestone is VTune profiling. We now have Intel GPU latency numbers for the DiT action head, so the next question is no longer:

```text
Can the model run on Intel hardware?
```

It is:

```text
Where is the runtime going, and what should we optimize first?
```

VTune is the tool that helps answer that.

## What VTune Is

Intel VTune Profiler is a performance analysis tool for Intel CPUs and GPUs. It records what the program and hardware are doing while a workload runs.

A normal benchmark gives one number:

```text
fused DiT on Intel Arc 140V GPU = 54.39 ms
```

VTune helps break that number down:

```text
54.39 ms =
  attention kernels
  matmul kernels
  normalization kernels
  elementwise kernels
  memory movement
  host/device synchronization
  dispatch overhead
  idle/stall time
```

That is why VTune is important before kernel work. Without profiling, we might optimize the wrong thing.

## What VTune Can Tell Us

For this project, useful VTune outputs include:

| Question | Why It Matters |
|---|---|
| Which kernels take the most time? | Identifies actual bottlenecks. |
| Is GPU utilization high or low? | Tells whether GPU is being kept busy. |
| Are execution units active or stalled? | Helps distinguish compute bottlenecks from memory/scheduling bottlenecks. |
| Is memory bandwidth saturated? | Tests whether the workload is memory-bound. |
| Are there many small kernels? | Indicates dispatch/fusion overhead. |
| Is there host/device synchronization? | Explains Python/runtime overhead or bubbles. |
| Does fused-loop reduce dispatch gaps? | Validates the fused-loop architecture beyond latency numbers. |
| Are MVN/LayerNorm-related kernels significant? | Determines whether AdaLayerNorm fusion is a good kernel target. |
| Is attention/matmul dominant? | May shift the optimization target toward SDPA/matmul/layout/precision. |

## CPU Hotspots vs GPU Hotspots

VTune has multiple analysis types. The most relevant are:

### CPU Hotspots

This profiles where CPU time is spent.

Useful for:

- Python orchestration overhead,
- OpenVINO host-side runtime overhead,
- model compile overhead,
- CPU plugin execution,
- synchronization between Python and OpenVINO.

### GPU Hotspots

This profiles GPU execution.

Useful for:

- GPU kernel timeline,
- GPU occupancy/utilization,
- EU activity,
- memory bandwidth,
- kernel duration,
- whether kernels are fragmented or fused.

For this project, GPU Hotspots is the main target because the best current DiT result is:

```text
Intel Arc 140V fused-loop DiT = 54.39 ms
```

## What We Are Profiling

We should profile at least two workloads:

### 1. Python-Orchestrated Single-Step Loop

```text
Python loop
  -> OpenVINO single-step DiT call
  -> Python action update
  -> repeat 4 times
```

Why:

- Shows overhead from repeated Python/OpenVINO calls.
- Helps explain why fused-loop is faster.
- Gives a baseline for dispatch/synchronization bubbles.

### 2. Fused 4-Step DiT IR

```text
one OpenVINO call
  -> unrolled 4-step DiT graph
  -> action chunk
```

Why:

- This is the current deployment candidate.
- Shows real GPU bottlenecks after orchestration overhead is reduced.
- Determines first kernel/runtime optimization target.

## What We Expect To See

Based on architecture, likely expensive sections are:

1. Transformer attention.
2. Feed-forward / MLP matmuls.
3. Cross-attention from action tokens to VLM embeddings.
4. AdaLayerNorm / MVN + scale + shift patterns.
5. Elementwise update/integration operations.

But the important point is that this is only a hypothesis. VTune should decide.

## AdaLayerNorm Hypothesis

The proposal targets AdaLayerNorm fusion because the pattern appears repeatedly inside the DiT blocks:

```text
LayerNorm / MVN
-> multiply by timestep-conditioned scale
-> add timestep-conditioned shift
```

Potential issue:

- If this remains decomposed into separate kernels, it may create repeated memory movement and dispatch overhead.

What VTune should confirm:

- Are MVN/normalization kernels visible and costly?
- Are there many small elementwise kernels around MVN?
- Does their total time justify a C++/GPU plugin contribution?

If AdaLayerNorm is not a major hotspot, we should not force that contribution just because it was in the proposal. We should use the profiling data.

## Attention / MatMul Hypothesis

The model is transformer-heavy, so attention and feed-forward matmuls may dominate.

VTune/OpenVINO profiling should answer:

- Are matmul kernels dominating GPU time?
- Is cross-attention more expensive than self-attention?
- Is fused SDPA being selected where expected?
- Are layouts causing extra reorders/transposes?
- Is precision FP32/FP16 affecting throughput?

If attention/matmul dominates, the next work may be:

- verifying SDPA fusion,
- changing precision hints,
- checking layout/reorder overhead,
- profiling GPU matmul utilization.

## What Counts As A Good Profiling Result

A useful VTune result should let us say:

```text
The top bottleneck is X.
It accounts for Y% of runtime.
It appears because of Z.
Therefore the next optimization should be A, not B.
```

Bad profiling result:

```text
VTune ran and produced a lot of data.
```

Good profiling result:

```text
Fused DiT GPU time is dominated by matmul/attention kernels.
AdaLayerNorm/MVN contributes only a small percentage, so kernel fusion should wait.
```

or:

```text
AdaLayerNorm/MVN and adjacent elementwise kernels create repeated small dispatches across the fused graph.
This validates the planned fusion target.
```

## Practical Profiling Checklist

### Setup

- [x] Add a stable profiling workload script: `export_tests/profile_dit_workload.py`.
- [x] Add a GitHub Actions VTune availability check.
- [x] Add an opt-in `run_vtune` workflow input for GPU Hotspots.
- [x] Check whether VTune is available on the Intel runner `PATH`.
- [ ] Locate or install VTune on the Intel runner.
- [ ] Run the workflow manually with `run_vtune=true`.
- [ ] If VTune is not installed, determine whether it can be installed during the reservation.
- [ ] Confirm command-line VTune works after install/path setup.
- [ ] Confirm GPU profiling is supported on this machine.

Current runner result:

```text
Workflow run: 27985830438
VTune CLI was not found on PATH.
```

Detailed note:

```text
notes/intel_runner_vtune_check_2026_06_23.md
```

## Implemented Profiling Harness

The profiling harness is:

```text
export_tests/profile_dit_workload.py
```

It creates the same synthetic DiT inputs used by the hardware benchmark:

```text
vl_embs:  [1, 512, 2048]
actions:  [1, 25, 23]
state:    [1, 1, 23]
timestep: [1]
```

It supports three modes:

```text
--mode fused
--mode python_loop
--mode both
```

This gives VTune a stable workload to profile without mixing profiling logic into the benchmark script.

Manual local/runner command:

```powershell
python export_tests\profile_dit_workload.py --device GPU --mode both --iterations 50 --warmup 5
```

Workflow usage:

1. Open the `Intel Hardware Benchmark` workflow.
2. Click `Run workflow`.
3. Set:

```text
run_vtune = true
vtune_iterations = 50
```

The workflow now checks for:

```text
vtune
amplxe-cl
```

If `run_vtune=true` and `vtune` is not on `PATH`, the VTune step fails clearly. Normal benchmark runs do not require VTune.

Expected VTune outputs:

```text
benchmark_outputs/vtune_gpu_fused/
benchmark_outputs/vtune_gpu_fused_summary.txt
benchmark_outputs/vtune_gpu_python_loop/
benchmark_outputs/vtune_gpu_python_loop_summary.txt
benchmark_outputs/dit_profile_fused.json
benchmark_outputs/dit_profile_python_loop.json
```

## Why This Design

The previous benchmark already answers:

```text
How fast is the DiT action head?
```

The profiling harness answers:

```text
Where does the time go during that workload?
```

Keeping the workload script separate from VTune has two benefits:

1. The same script can be run without VTune to confirm it works.
2. VTune can wrap exactly one stable Python process, which makes the trace easier to interpret.

### Workloads

- [x] Create a benchmark mode that runs only fused-loop GPU repeatedly.
- [x] Create a benchmark mode that runs only Python-loop GPU repeatedly.
- [x] Add configurable iteration count for VTune collection.
- [x] Add interim OpenVINO per-node profiling while VTune is unavailable.
- [ ] Tune run duration enough for VTune to collect stable data.
- [ ] Avoid mixing conversion/compile time into runtime profiling unless intentionally measuring compile.
- [x] Save output directories as GitHub Actions artifacts or local zip files.

## Interim OpenVINO Node Profiling

Because VTune is not currently available on the Intel runner `PATH`, we added:

```text
export_tests/openvino_node_profile.py
```

This script compiles the DiT IR with:

```text
PERF_COUNT = YES
```

and writes:

```text
benchmark_outputs/openvino_node_profile.json
benchmark_outputs/openvino_node_profile.md
```

It profiles two modes on GPU:

1. `fused_loop_4_step`
2. `python_loop_4_step`

This is not a replacement for VTune GPU Hotspots because it does not expose the same low-level EU utilization, memory bandwidth, or hardware stall metrics. It is still useful because it can identify which OpenVINO graph node types and node names dominate runtime.

Use this result to guide the next question:

```text
Do MVN/AdaLayerNorm-like nodes show up as meaningful runtime, or is runtime dominated by MatMul/attention?
```

### Metrics To Capture

- [ ] Total runtime.
- [ ] GPU kernel timeline.
- [ ] Top kernels by time.
- [ ] GPU utilization.
- [ ] EU active/stall/idle metrics if available.
- [ ] Memory bandwidth.
- [ ] Kernel launch count.
- [ ] Host/device synchronization points.
- [ ] CPU-side OpenVINO overhead.

### Analysis

- [ ] Compare Python-loop vs fused-loop.
- [ ] Identify whether fused-loop reduces dispatch gaps.
- [ ] Identify whether attention/matmul dominates.
- [ ] Identify MVN/AdaLayerNorm contribution.
- [ ] Check for reorder/transpose/layout overhead.
- [ ] Decide optimization target from evidence.

### Documentation

- [ ] Save raw VTune results.
- [ ] Export summary tables/screenshots if possible.
- [ ] Document command used.
- [ ] Document machine/device/runtime versions.
- [ ] Write interpretation in `notes/`.
- [ ] Convert key findings into presentation slides.

## Possible Command-Line Direction

Exact commands depend on the installed VTune version. The command-line tool is often named one of:

```text
vtune
vtune-gui
amplxe-cl
```

The general pattern is:

```powershell
vtune -collect gpu-hotspots -result-dir benchmark_outputs\vtune_gpu_fused -- python export_tests\profile_dit_workload.py --mode fused --device GPU --iterations 50
```

and:

```powershell
vtune -collect gpu-hotspots -result-dir vtune_results\python_loop_gpu -- python export_tests\run_profile_target.py --mode python-loop --device GPU --runs 200
```

We may need to create `run_profile_target.py` so the profiling workload is simple, repeatable, and does not include conversion or setup noise.

## Immediate Next Technical Step

Before running VTune, create a small profiling target script that:

1. Loads existing `single_step_dit.xml` and `fused_loop_dit.xml`.
2. Compiles only for `GPU`.
3. Warms up.
4. Runs either:
   - fused-loop only, or
   - Python-loop only.
5. Runs long enough for profiling.

This avoids profiling unrelated setup like dependency install, IR conversion, or GitHub Actions checkout.

## Presentation Angle

For mentors, the story should be:

1. We now have real Intel GPU latency.
2. Fused-loop is the better deployment graph for the DiT action head.
3. The next question is no longer whether it runs, but what dominates the GPU runtime.
4. VTune will determine whether the first contribution should target AdaLayerNorm/MVN, attention/matmul, layout/reorder, or runtime orchestration.
