from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov


REPO_ROOT = Path(__file__).resolve().parents[1]
IR_DIR = REPO_ROOT / "artifacts" / "openvino_ir"
OUT_DIR = REPO_ROOT / "benchmark_outputs"

VLM_IR = IR_DIR / os.environ.get("VLM_IR_NAME", "qwen_vlm_backbone_from_onnx.xml")
FUSED_DIT_IR = IR_DIR / os.environ.get("FUSED_DIT_IR_NAME", "fused_loop_dit.xml")

DEFAULT_ACTION_HORIZON = int(os.environ.get("VLA_ACTION_HORIZON", "25"))
DEFAULT_ACTION_DIM = int(os.environ.get("VLA_ACTION_DIM", "23"))
DEFAULT_STATE_DIM = int(os.environ.get("VLA_STATE_DIM", "23"))
NPU_ENABLED = os.environ.get("BENCHMARK_NPU", "0") == "1"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo))


def stats(samples: list[float]) -> dict:
    return {
        "runs": len(samples),
        "mean_ms": float(statistics.mean(samples)),
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "std_ms": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        "p90_ms": percentile(samples, 0.90),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
    }


def safe_name(port, fallback: str) -> str:
    try:
        return port.get_any_name()
    except RuntimeError:
        return fallback


def partial_rank(port) -> int:
    rank = port.get_partial_shape().rank
    if rank.is_dynamic:
        raise RuntimeError(f"Dynamic-rank input is not supported: {safe_name(port, '<unnamed>')}")
    return int(rank.get_length())


def concrete_shape(port) -> list[int]:
    shape = port.get_partial_shape()
    if shape.is_static:
        return [int(dim.get_length()) for dim in shape]
    name = safe_name(port, "")
    if name in {"input_ids", "attention_mask"}:
        return [1, int(os.environ.get("VLM_TEXT_TOKENS", "92"))]
    if "Cast_output" in name or name == "pixel_values":
        return [int(os.environ.get("VLM_IMAGE_TOKENS", "256")), int(os.environ.get("VLM_PIXEL_DIM", "1176"))]
    if name == "image_grid_thw":
        return [1, 3]
    raise RuntimeError(f"Cannot choose concrete shape for {name}: {shape}")


def numpy_dtype(element_type) -> np.dtype:
    text = str(element_type)
    if "float16" in text:
        return np.float16
    if "float32" in text:
        return np.float32
    if "int64" in text:
        return np.int64
    if "int32" in text:
        return np.int32
    if "boolean" in text or "bool" in text:
        return np.bool_
    raise RuntimeError(f"Unsupported input element type: {element_type}")


def make_vlm_input(port, index: int) -> np.ndarray:
    shape = concrete_shape(port)
    dtype = numpy_dtype(port.get_element_type())
    name = safe_name(port, f"input_{index}")
    if np.issubdtype(dtype, np.floating):
        rng = np.random.default_rng(4200 + index)
        return rng.standard_normal(shape).astype(dtype)
    if name == "attention_mask":
        return np.ones(shape, dtype=dtype)
    if name == "image_grid_thw" and shape == [1, 3]:
        return np.array([[1, 16, 16]], dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def make_dit_static_inputs(vl_embs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4300)
    batch = int(vl_embs.shape[0])
    initial_noise = rng.standard_normal(
        (batch, DEFAULT_ACTION_HORIZON, DEFAULT_ACTION_DIM),
        dtype=np.float32,
    )
    state = rng.standard_normal((batch, 1, DEFAULT_STATE_DIM), dtype=np.float32)
    return initial_noise, state


def map_fused_dit_inputs(compiled: ov.CompiledModel, vl_embs, initial_noise, state) -> dict:
    rank3_inputs = [port for port in compiled.inputs if partial_rank(port) == 3]
    if len(rank3_inputs) != 3:
        raise RuntimeError(f"Expected 3 rank-3 fused DiT inputs, got {len(rank3_inputs)}")

    values = {
        "vl_embs": vl_embs.astype(np.float32, copy=False),
        "initial_noise": initial_noise,
        "state": state,
    }
    shapes = {name: value.shape for name, value in values.items()}

    mapped = {}
    used = set()
    for port in rank3_inputs:
        pshape = port.get_partial_shape()
        static_shape = [int(pshape[i].get_length()) for i in range(3)] if pshape.is_static else None
        candidates = [
            name
            for name, shape in shapes.items()
            if name not in used and (static_shape is None or list(shape) == static_shape)
        ]
        if len(candidates) != 1:
            # Fall back to naming conventions from the export wrappers.
            pname = safe_name(port, "")
            if "vl_emb" in pname:
                candidates = ["vl_embs"]
            elif "state" in pname:
                candidates = ["state"]
            elif "noise" in pname or "action" in pname:
                candidates = ["initial_noise"]
        if len(candidates) != 1 or candidates[0] in used:
            raise RuntimeError(
                f"Could not map DiT input {safe_name(port, '<unnamed>')} shape={port.get_partial_shape()} "
                f"against values {shapes}"
            )
        role = candidates[0]
        mapped[port] = values[role]
        used.add(role)
    return mapped


def timed_call(fn):
    start = time.perf_counter()
    value = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, value


def measure(fn, warmup: int, runs: int) -> dict:
    warmup_samples = [timed_call(fn)[0] for _ in range(warmup)]
    samples = [timed_call(fn)[0] for _ in range(runs)]
    result = stats(samples)
    result["warmup_runs"] = warmup
    result["warmup_mean_ms"] = float(statistics.mean(warmup_samples)) if warmup_samples else 0.0
    result["first_warmup_ms"] = float(warmup_samples[0]) if warmup_samples else None
    return result


def compile_with_time(core: ov.Core, model: ov.Model, device: str):
    start = time.perf_counter()
    compiled = core.compile_model(model, device)
    return compiled, (time.perf_counter() - start) * 1000


def benchmark_device(core: ov.Core, vlm_model: ov.Model, dit_model: ov.Model, device: str) -> dict:
    if device == "NPU" and not NPU_ENABLED:
        return {"device": device, "status": "skipped", "reason": "NPU disabled by default."}

    runs = int(os.environ.get(f"E2E_BENCHMARK_RUNS_{device}", "20" if device == "GPU" else "3"))
    warmup = int(os.environ.get(f"E2E_BENCHMARK_WARMUP_{device}", "5" if device == "GPU" else "1"))

    vlm_compiled, vlm_compile_ms = compile_with_time(core, vlm_model, device)
    dit_compiled, dit_compile_ms = compile_with_time(core, dit_model, device)

    vlm_inputs = {port: make_vlm_input(port, i) for i, port in enumerate(vlm_compiled.inputs)}
    first_vlm_outputs = vlm_compiled(vlm_inputs)
    vl_embs = next(iter(first_vlm_outputs.values()))
    initial_noise, state = make_dit_static_inputs(vl_embs)
    dit_inputs = map_fused_dit_inputs(dit_compiled, vl_embs, initial_noise, state)

    def run_vlm():
        return next(iter(vlm_compiled(vlm_inputs).values()))

    def run_dit_static():
        return dit_compiled(dit_inputs)

    def run_e2e():
        vlm_outputs = vlm_compiled(vlm_inputs)
        current_vl_embs = next(iter(vlm_outputs.values()))
        current_noise, current_state = make_dit_static_inputs(current_vl_embs)
        current_dit_inputs = map_fused_dit_inputs(
            dit_compiled,
            current_vl_embs,
            current_noise,
            current_state,
        )
        return dit_compiled(current_dit_inputs)

    return {
        "device": device,
        "status": "ok",
        "runs": runs,
        "warmup": warmup,
        "vlm_compile_ms": vlm_compile_ms,
        "dit_compile_ms": dit_compile_ms,
        "vlm_output_shape": list(vl_embs.shape),
        "dit_output_shape": list(next(iter(dit_compiled(dit_inputs).values())).shape),
        "vlm_latency": measure(run_vlm, warmup=warmup, runs=runs),
        "dit_latency_with_vlm_output": measure(run_dit_static, warmup=warmup, runs=runs),
        "end_to_end_latency": measure(run_e2e, warmup=warmup, runs=runs),
    }


def port_metadata(model: ov.Model) -> dict:
    return {
        "inputs": [
            {
                "name": safe_name(port, f"input_{i}"),
                "shape": str(port.get_partial_shape()),
                "type": str(port.get_element_type()),
            }
            for i, port in enumerate(model.inputs)
        ],
        "outputs": [
            {
                "name": safe_name(port, f"output_{i}"),
                "shape": str(port.get_partial_shape()),
                "type": str(port.get_element_type()),
            }
            for i, port in enumerate(model.outputs)
        ],
    }


def write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# End-to-End OpenVINO VLA Benchmark",
        "",
        f"OpenVINO: `{results['openvino_version']}`",
        f"VLM IR: `{results['vlm_ir']}`",
        f"Fused DiT IR: `{results['fused_dit_ir']}`",
        "",
        "This measures the OpenVINO model chain: VLM IR inference, tensor handoff in Python, and fused DiT IR inference. It does not include Qwen processor/image preprocessing.",
        "",
        "| Device | VLM Mean | DiT Mean | End-to-End Mean | End-to-End P95 | VLM Output | DiT Output | Status |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for item in results["device_results"]:
        if item.get("status") != "ok":
            lines.append(
                f"| {item['device']} | n/a | n/a | n/a | n/a | n/a | n/a | {item.get('status')}: {item.get('reason', item.get('error', ''))} |"
            )
            continue
        lines.append(
            "| {device} | {vlm:.2f} ms | {dit:.2f} ms | {e2e:.2f} ms | {p95:.2f} ms | `{vshape}` | `{dshape}` | ok |".format(
                device=item["device"],
                vlm=item["vlm_latency"]["mean_ms"],
                dit=item["dit_latency_with_vlm_output"]["mean_ms"],
                e2e=item["end_to_end_latency"]["mean_ms"],
                p95=item["end_to_end_latency"]["p95_ms"],
                vshape=item["vlm_output_shape"],
                dshape=item["dit_output_shape"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    if not VLM_IR.exists() or not VLM_IR.with_suffix(".bin").exists():
        raise FileNotFoundError(f"Missing VLM IR pair: {VLM_IR}")
    if not FUSED_DIT_IR.exists() or not FUSED_DIT_IR.with_suffix(".bin").exists():
        raise FileNotFoundError(f"Missing fused DiT IR pair: {FUSED_DIT_IR}")

    core = ov.Core()
    vlm_model = core.read_model(VLM_IR)
    dit_model = core.read_model(FUSED_DIT_IR)
    results = {
        "openvino_version": ov.__version__,
        "available_devices": core.available_devices,
        "vlm_ir": str(VLM_IR),
        "fused_dit_ir": str(FUSED_DIT_IR),
        "vlm_model": port_metadata(vlm_model),
        "fused_dit_model": port_metadata(dit_model),
        "device_results": [],
    }

    for device in core.available_devices:
        print(f"\n=== End-to-end VLA benchmark: {device} ===")
        try:
            item = benchmark_device(core, vlm_model, dit_model, device)
        except Exception as exc:
            item = {"device": device, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(item, indent=2))
        results["device_results"].append(item)

    json_path = OUT_DIR / "end_to_end_vla_openvino_benchmark.json"
    md_path = OUT_DIR / "end_to_end_vla_openvino_benchmark.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, md_path)
    print(f"\n[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {md_path}")

    if not any(item.get("status") == "ok" for item in results["device_results"]):
        raise RuntimeError("No devices produced end-to-end VLA benchmark results")


if __name__ == "__main__":
    main()
