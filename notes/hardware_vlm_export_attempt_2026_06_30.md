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

## Presentation Wording

Short version:

"I moved the VLM blocker forward. It is no longer just a missing artifact. I tried the real UnifoLM VLM checkpoint, made the export path configurable, removed CUDA-only assumptions, and generated valid Qwen processor inputs. The real export now fails in OpenVINO conversion of Qwen2.5-VL's visual window-indexing operations, specifically unsupported `aten::index`, `aten::unbind`, and list-unpack patterns. I am not reporting fake full-VLA latency from the tiny placeholder artifact because that would be misleading."
