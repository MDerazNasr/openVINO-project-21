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


def timed_call(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def measure(fn, warmup: int, runs: int) -> dict:
    warmup_samples = [timed_call(fn) for _ in range(warmup)]
    samples = [timed_call(fn) for _ in range(runs)]
    result = stats(samples)
    result["warmup_runs"] = warmup
    result["warmup_mean_ms"] = float(statistics.mean(warmup_samples)) if warmup_samples else 0.0
    result["first_warmup_ms"] = float(warmup_samples[0]) if warmup_samples else None
    return result


def safe_name(port, fallback: str) -> str:
    try:
        return port.get_any_name()
    except RuntimeError:
        return fallback


def concrete_shape(port) -> list[int]:
    shape = port.get_partial_shape()
    if shape.is_static:
        return [int(dim.get_length()) for dim in shape]

    name = safe_name(port, "")
    rank = shape.rank.get_length()
    if name in {"input_ids", "attention_mask"}:
        return [1, int(os.environ.get("VLM_TEXT_TOKENS", "35"))]
    if name == "pixel_values":
        return [int(os.environ.get("VLM_IMAGE_TOKENS", "256")), int(os.environ.get("VLM_PIXEL_DIM", "1176"))]
    if name == "image_grid_thw":
        return [1, 3]
    if rank == 1:
        return [1]
    if rank == 2:
        return [1, 1]
    if rank == 3:
        return [1, 1, 1]
    raise RuntimeError(f"Cannot choose concrete benchmark shape for input {name} with shape {shape}")


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
    raise RuntimeError(f"Unsupported input element type for benchmark: {element_type}")


def make_input(port, index: int) -> np.ndarray:
    shape = concrete_shape(port)
    dtype = numpy_dtype(port.get_element_type())
    name = safe_name(port, f"input_{index}")

    if np.issubdtype(dtype, np.floating):
        rng = np.random.default_rng(42 + index)
        return rng.standard_normal(shape).astype(dtype)
    if name == "attention_mask":
        return np.ones(shape, dtype=dtype)
    if name == "image_grid_thw" and shape == [1, 3]:
        return np.array([[1, 16, 16]], dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def model_metadata(model: ov.Model) -> dict:
    return {
        "inputs": [
            {
                "index": i,
                "name": safe_name(port, f"input_{i}"),
                "shape": str(port.get_partial_shape()),
                "chosen_shape": concrete_shape(port),
                "type": str(port.get_element_type()),
            }
            for i, port in enumerate(model.inputs)
        ],
        "outputs": [
            {
                "index": i,
                "name": safe_name(port, f"output_{i}"),
                "shape": str(port.get_partial_shape()),
                "type": str(port.get_element_type()),
            }
            for i, port in enumerate(model.outputs)
        ],
        "op_count": len(model.get_ops()),
    }


def benchmark_device(core: ov.Core, model: ov.Model, device: str) -> dict:
    if device == "NPU" and not NPU_ENABLED:
        return {
            "device": device,
            "status": "skipped",
            "reason": "NPU disabled by default until CPU/GPU VLM baselines are stable.",
        }

    runs = int(os.environ.get(f"VLM_BENCHMARK_RUNS_{device}", "30" if device == "GPU" else "5"))
    warmup = int(os.environ.get(f"VLM_BENCHMARK_WARMUP_{device}", "5" if device == "GPU" else "2"))

    compile_start = time.perf_counter()
    compiled = core.compile_model(model, device)
    compile_ms = (time.perf_counter() - compile_start) * 1000

    inputs = {port: make_input(port, i) for i, port in enumerate(compiled.inputs)}

    def run_once():
        return compiled(inputs)

    return {
        "device": device,
        "status": "ok",
        "runs": runs,
        "warmup": warmup,
        "compile_ms": compile_ms,
        "latency": measure(run_once, warmup=warmup, runs=runs),
    }


def write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# Qwen VLM OpenVINO Benchmark",
        "",
        f"OpenVINO: `{results['openvino_version']}`",
        f"IR: `{results['ir']['xml']}`",
        f"XML bytes: `{results['ir']['xml_bytes']}`",
        f"BIN bytes: `{results['ir']['bin_bytes']}`",
        "",
        "## Inputs",
        "",
        "| Name | IR Shape | Benchmark Shape | Type |",
        "|---|---|---|---|",
    ]
    for item in results["model"]["inputs"]:
        lines.append(f"| {item['name']} | `{item['shape']}` | `{item['chosen_shape']}` | `{item['type']}` |")

    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Device | Mean | Median | P95 | Compile | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in results["device_results"]:
        if item.get("status") != "ok":
            lines.append(f"| {item['device']} | n/a | n/a | n/a | n/a | {item.get('status')}: {item.get('reason', '')} |")
            continue
        latency = item["latency"]
        lines.append(
            "| {device} | {mean:.2f} ms | {median:.2f} ms | {p95:.2f} ms | {compile:.2f} ms | ok |".format(
                device=item["device"],
                mean=latency["mean_ms"],
                median=latency["median_ms"],
                p95=latency["p95_ms"],
                compile=item["compile_ms"],
            )
        )

    lines.extend(
        [
            "",
            "This benchmark measures the exported Qwen/UnifoLM VLM backbone only. Full VLA latency still requires connecting this output to the DiT action head and validating the handoff shape/semantics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    core = ov.Core()

    bin_path = VLM_IR.with_suffix(".bin")
    if not VLM_IR.exists():
        raise FileNotFoundError(f"Missing VLM IR XML: {VLM_IR}")
    if not bin_path.exists():
        raise FileNotFoundError(f"Missing VLM IR BIN: {bin_path}")

    model = core.read_model(VLM_IR)
    results = {
        "openvino_version": ov.__version__,
        "available_devices": core.available_devices,
        "ir": {
            "xml": str(VLM_IR),
            "xml_bytes": VLM_IR.stat().st_size,
            "bin": str(bin_path),
            "bin_bytes": bin_path.stat().st_size,
        },
        "model": model_metadata(model),
        "device_results": [],
    }

    for device in core.available_devices:
        print(f"\n=== VLM benchmark: {device} ===")
        try:
            item = benchmark_device(core, model, device)
        except Exception as exc:
            item = {"device": device, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(item, indent=2))
        results["device_results"].append(item)

    json_path = OUT_DIR / "qwen_vlm_openvino_benchmark.json"
    md_path = OUT_DIR / "qwen_vlm_openvino_benchmark.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, md_path)
    print(f"\n[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {md_path}")

    if not any(item.get("status") == "ok" for item in results["device_results"]):
        raise RuntimeError("No devices produced VLM benchmark results")


if __name__ == "__main__":
    main()
