# Hardware VLM Export Attempt - 2026-06-30

## Goal

Unblock the full VLM/full VLA benchmark requested after the DiT action-head benchmark was validated on Intel CPU/GPU hardware.

The earlier benchmark was intentionally scoped to the DiT action head with synthetic VLM embeddings. Full end-to-end VLA latency requires a real Qwen2.5-VL/OpenVINO artifact with weights, not the tiny placeholder artifact in `artifacts/openvino_ir`.

## Starting State

- `qwen_vlm_backbone.xml` existed but was only a tiny/template artifact.
- `qwen_vlm_backbone.bin` was missing or effectively empty, so it could not support real VLM latency claims.
- `unifolm_vla_train.yaml` pointed to a local training path:
  - `/root/unitree_jiang/Unifolm-VLM-0`
- The upstream Qwen wrapper assumed CUDA/FlashAttention:
  - `attn_implementation="flash_attention_2"`
  - `torch_dtype=torch.bfloat16`
  - `device_map="cuda"`
- The Intel runner is a Windows machine with OpenVINO CPU/GPU/NPU devices and CPU PyTorch, not a CUDA environment.

## What We Changed

### Real Checkpoint Selection

Added configurable VLM checkpoint selection to `export_tests/convert_qwen_vlm.py`.

The workflow now accepts:

- `base_vlm`
- `attn_implementation`
- `allow_mock`

For the real export attempt, we used:

- `base_vlm=unitreerobotics/Unifolm-VLM-Base`
- `attn_implementation=eager`
- `allow_mock=false`

This replaced the invalid local `/root/...` path with a public HuggingFace checkpoint.

### CPU/Export-Friendly Qwen Wrapper

Added a vendor patch for:

`export_tests/vendor_patches/unifolm_vla/model/modules/vlm/QWen2_5.py`

The patch removes hardcoded CUDA assumptions by making attention implementation, dtype, and device map configurable. It also avoids CUDA autocast when CUDA is unavailable.

This was necessary because the GitHub runner can execute on the Intel hardware but does not expose CUDA.

### Real Processor Inputs

The first real export attempt loaded the checkpoint but failed because the script passed fake raw image tensors:

- `pixel_values=(1, 3, 224, 224)`
- `image_grid_thw=(1, 28, 28)`

Qwen2.5-VL expects processor-packed image patches, not raw image tensors in that path. We changed the script to generate inputs through the real Qwen processor:

- `input_ids shape: (1, 92)`
- `attention_mask shape: (1, 92)`
- `pixel_values shape: (256, 1176)`
- `image_grid_thw shape: (1, 3)`

This fixed the earlier image-grid mismatch.

## Hardware Workflow Runs

### Run `28438861026`

Result: failed after real checkpoint load reached tracing.

Primary blocker:

`IndexError: index 40 is out of bounds for dimension 0 with size 32`

Cause:

The example image input did not match Qwen2.5-VL's expected processed patch/grid representation.

Decision:

Use the model's processor to generate valid multimodal example inputs instead of manually inventing raw pixel tensors.

### Run `28445925313`

Result: failed after loading the real checkpoint and tracing with valid processor inputs.

Important progress:

- The public UnifoLM VLM checkpoint was found and loaded.
- The CPU/export-friendly wrapper was applied.
- The invalid dummy image-grid issue was fixed.
- The conversion reached OpenVINO frontend conversion of the real Qwen visual path.

Primary blocker:

OpenVINO PyTorch frontend could not fully convert Qwen2.5-VL visual windowing/indexing operations:

- `aten::index`
- `aten::unbind`
- `prim::ListUnpack`
- `SequenceMark`

The failure occurs in the Qwen2.5-VL visual transformer window-indexing/reordering path, around the visual model's `window_index` logic.

## Why We Did Not Fake The Full VLM Benchmark

The existing tiny VLM artifact is not a real weighted Qwen2.5-VL export. Reporting its latency would make the full VLA benchmark look better but would be technically wrong.

Likewise, removing or bypassing Qwen's visual window-indexing logic just to make conversion pass would change model semantics. That would no longer be a faithful full VLM benchmark.

So the correct current statement is:

- DiT OpenVINO hardware benchmarking is valid.
- Real VLM checkpoint loading now works.
- Full Qwen2.5-VL OpenVINO export is blocked by unsupported PyTorch frontend conversion patterns in the visual path.
- Full end-to-end VLA latency remains blocked until the real VLM visual path is exported or a supported OpenVINO model path is used.

## Next Options

1. Ask mentors/OpenVINO maintainers whether Qwen2.5-VL visual export has a recommended path, existing notebook, model optimizer path, or known patch for the visual window-indexing logic.
2. Try a split benchmark:
   - VLM language/backbone transformer with precomputed/synthetic visual embeddings.
   - DiT action head with real or synthetic VLM embeddings.
   - Keep visual encoder export listed as blocked.
3. Try alternate export routes:
   - ONNX export followed by OpenVINO conversion.
   - `optimum-intel`/OpenVINO model export tools if they support Qwen2.5-VL.
4. Try a local/static-shape patch of Qwen visual window-indexing only if it preserves semantics for fixed image size.
5. Continue using the GitHub runner tunnel while SSH remains unresolved.

## ONNX Fallback Update

After the direct PyTorch-to-OpenVINO path failed in Qwen2.5-VL visual indexing/list operations, we tested ONNX as an alternate bridge.

What changed:

- Added `export_tests/export_qwen_vlm_onnx.py`.
- Added `.github/workflows/vlm-onnx-export-check.yml`.
- Installed `onnx` and `onnxscript`.
- Forced the legacy ONNX tracer with `dynamo=False` after the newer Torch ONNX exporter failed on data-dependent `grid_thw` integer extraction in `get_vision_position_ids`.
- Set UTF-8 logging variables on Windows to avoid console encoding failures.

Result:

- The real UnifoLM VLM checkpoint loaded.
- The processor-backed Qwen multimodal inputs were used.
- ONNX export succeeded.
- ONNX inspection succeeded.

Operational issue found:

- Uploading the full ONNX artifact through GitHub Actions was too slow/large and left the self-hosted runner stuck in the upload step.
- The runner was recovered by stopping the stale runner session and reconnecting it.
- The workflow was changed to convert ONNX to OpenVINO IR on the runner, benchmark the generated IR in the same job, and upload only small benchmark reports.

This is important because ONNX export is now a viable route around the direct OpenVINO PyTorch frontend blocker.

## ONNX-to-OpenVINO VLM Benchmark Update

Workflow run:

- `VLM ONNX Export Check`
- Run: `https://github.com/MDerazNasr/openVINO-project-21/actions/runs/28464075911`
- Artifact: `https://github.com/MDerazNasr/openVINO-project-21/actions/runs/28464075911/artifacts/7989199564`

What passed:

- Real UnifoLM VLM checkpoint loaded from `unitreerobotics/Unifolm-VLM-Base`.
- ONNX export passed.
- ONNX inspection passed.
- ONNX-to-OpenVINO conversion passed.
- OpenVINO IR inspection passed.
- VLM-only OpenVINO benchmark passed on CPU and GPU.

Generated VLM IR:

- XML bytes: `4,844,170`
- BIN bytes: `15,494,478,376`

Benchmark input shapes:

| Input | Shape | Type |
|---|---:|---|
| `input_ids` | `[1, 92]` | `int64` |
| `attention_mask` | `[1, 92]` | `int64` |
| pixel tensor | `[256, 1176]` | `float32` |

Output shape:

- `last_hidden_state`: `[1, 92, 3584]`

Latency:

| Device | Mean | Median | P95 | Compile | Status |
|---|---:|---:|---:|---:|---|
| CPU | `7224.91 ms` | `3748.25 ms` | `17543.91 ms` | `28315.98 ms` | ok |
| GPU | `207.07 ms` | `206.18 ms` | `209.55 ms` | `52371.15 ms` | ok |
| NPU | n/a | n/a | n/a | n/a | skipped |

Interpretation:

- We now have a real VLM-only hardware benchmark.
- The VLM is much heavier than the fused DiT action head on GPU (`~207 ms` VLM vs `~54 ms` fused DiT), matching the mentor's expectation that the VLM may dominate full VLA latency.
- The CPU VLM benchmark has high variance and should be treated as a rough baseline; the GPU result is much more stable.
- Full end-to-end VLA latency is now closer, but still needs a validated handoff from the VLM output to the DiT `vl_embs` input. The exported VLM output is `[1, 92, 3584]`, while the current DiT benchmark fixture uses synthetic `vl_embs` shaped `[1, 512, 2048]`. We need to identify the real projection/selection step between VLM hidden states and DiT conditioning before claiming full VLA latency.
- We should not simply add the two numbers until we verify output shape/semantics and data movement.

## VLM-Compatible DiT Benchmark Update

Workflow run:

- `Intel Hardware Benchmark`
- Run: `https://github.com/MDerazNasr/openVINO-project-21/actions/runs/28466447460`
- Artifact: `https://github.com/MDerazNasr/openVINO-project-21/actions/runs/28466447460/artifacts/7989865577`

Why this run was needed:

- The original DiT benchmark used synthetic `vl_embs` shaped `[1, 512, 2048]`.
- The real VLM IR outputs `last_hidden_state` shaped `[1, 92, 3584]`.
- The standalone DiT converters bypass the full `Unifolm_VLA` framework, so they do not automatically set `cross_attention_dim` from the Qwen model hidden size.
- We added configurable `VLA_VL_SEQ_LEN` and `VLA_VL_DIM` support and forced DiT IR regeneration with `vl_seq_len=92`, `vl_dim=3584`.

Generated DiT IR:

- VLM conditioning shape: `seq_len=92`, `hidden_dim=3584`
- DiT parameters: `588,135,424`
- Single-step `.bin`: `1,199,096,776` bytes
- Fused-loop `.bin`: `1,199,096,816` bytes

VLM-compatible DiT latency:

| Device | Single Step Mean | Python Loop Mean | Fused Loop Mean | Fused Speedup | Fused Chunks/s | Status |
|---|---:|---:|---:|---:|---:|---|
| CPU | `224.95 ms` | `897.70 ms` | `770.06 ms` | `1.17x` | `1.30` | ok |
| GPU | `14.77 ms` | `59.06 ms` | `52.45 ms` | `1.13x` | `19.07` | ok |
| NPU | n/a | n/a | n/a | n/a | n/a | skipped |

Component-level GPU estimate:

- VLM-only GPU mean: `207.07 ms`
- VLM-compatible fused DiT GPU mean: `52.45 ms`
- Simple component sum: `~259.52 ms`

Important caveat:

- This is not yet a true end-to-end VLA runtime because we have not validated the exact runtime handoff, memory ownership, or semantic preprocessing between the VLM output and DiT input.
- It is a useful component-level estimate and strongly suggests the VLM dominates total latency.
- The optional node-profiling step failed because `profile_dit_workload.py` still had the old `2048` fixture hardcoded. The benchmark itself passed; profiling was patched afterward to use `VLA_VL_SEQ_LEN` and `VLA_VL_DIM`.

## Presentation Wording

Short version:

"I moved the VLM blocker forward. It is no longer just a missing artifact. The direct PyTorch-to-OpenVINO conversion path failed on Qwen2.5-VL visual indexing/list operations, so I used ONNX as a bridge. With the real UnifoLM VLM checkpoint, ONNX export now succeeds, ONNX-to-OpenVINO conversion succeeds, and the real VLM IR benchmarks on Intel CPU/GPU. The GPU VLM latency is about 207 ms, which suggests the VLM is likely the dominant part of end-to-end VLA latency compared with the fused DiT action head at about 54 ms. I am still not claiming full end-to-end VLA latency until I validate the VLM-output-to-DiT-input handoff."
