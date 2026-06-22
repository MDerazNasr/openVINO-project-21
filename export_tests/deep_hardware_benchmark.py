from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import openvino as ov


REPO_ROOT = Path(__file__).resolve().parents[1]
IR_DIR = REPO_ROOT / "artifacts" / "openvino_ir"
OUT_DIR = REPO_ROOT / "benchmark_outputs"

SINGLE_IR = IR_DIR / "single_step_dit.xml"
FUSED_IR = IR_DIR / "fused_loop_dit.xml"
VLM_IR = IR_DIR / "qwen_vlm_backbone.xml"

DEFAULT_BATCH = 1
DEFAULT_SEQ_LEN = 512
DEFAULT_VL_DIM = 2048
DEFAULT_ACTION_HORIZON = int(os.environ.get("VLA_ACTION_HORIZON", "25"))
DEFAULT_ACTION_DIM = int(os.environ.get("VLA_ACTION_DIM", "23"))
DEFAULT_STATE_DIM = int(os.environ.get("VLA_STATE_DIM", "23"))
NPU_ENABLED = os.environ.get("BENCHMARK_NPU", "0") == "1"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
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


def file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def model_info(core: ov.Core, xml_path: Path) -> dict:
    bin_path = xml_path.with_suffix(".bin")
    info = {
        "xml": file_info(xml_path),
        "bin": file_info(bin_path),
        "is_real_weight_artifact": bin_path.exists() and bin_path.stat().st_size > 100_000_000,
    }
    if not xml_path.exists():
        return info
    if not info["is_real_weight_artifact"]:
        info["status"] = "metadata_only"
        info["reason"] = "Missing or tiny .bin file; skipping OpenVINO model load."
        return info

    model = core.read_model(xml_path)
    ops = Counter(op.get_type_name() for op in model.get_ops())
    info["inputs"] = [
        {
            "index": i,
            "name": safe_any_name(port, f"input_{i}"),
            "shape": str(port.get_partial_shape()),
            "type": str(port.get_element_type()),
        }
        for i, port in enumerate(model.inputs)
    ]
    info["outputs"] = [
        {
            "index": i,
            "name": safe_any_name(port, f"output_{i}"),
            "shape": str(port.get_partial_shape()),
            "type": str(port.get_element_type()),
        }
        for i, port in enumerate(model.outputs)
    ]
    info["op_count"] = int(sum(ops.values()))
    info["op_types"] = dict(sorted(ops.items(), key=lambda item: (-item[1], item[0])))
    return info


def safe_any_name(port, fallback: str) -> str:
    try:
        return port.get_any_name()
    except RuntimeError:
        return fallback


def make_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    vl_embs = rng.standard_normal((DEFAULT_BATCH, DEFAULT_SEQ_LEN, DEFAULT_VL_DIM), dtype=np.float32)
    actions = rng.standard_normal((DEFAULT_BATCH, DEFAULT_ACTION_HORIZON, DEFAULT_ACTION_DIM), dtype=np.float32)
    state = rng.standard_normal((DEFAULT_BATCH, 1, DEFAULT_STATE_DIM), dtype=np.float32)
    timestep = np.zeros((DEFAULT_BATCH,), dtype=np.int64)
    return vl_embs, actions, state, timestep


def ports_by_rank(compiled: ov.CompiledModel, rank: int):
    return [port for port in compiled.inputs if port.get_partial_shape().rank.get_length() == rank]


def single_inputs(compiled: ov.CompiledModel, vl_embs, actions, state, timestep) -> dict:
    rank3 = ports_by_rank(compiled, 3)
    rank1 = ports_by_rank(compiled, 1)
    if len(rank3) != 3 or len(rank1) != 1:
        raise RuntimeError(f"Unexpected single-step inputs: rank3={len(rank3)}, rank1={len(rank1)}")
    return {rank3[0]: vl_embs, rank3[1]: actions, rank3[2]: state, rank1[0]: timestep}


def fused_inputs(compiled: ov.CompiledModel, vl_embs, actions, state) -> dict:
    rank3 = ports_by_rank(compiled, 3)
    if len(rank3) != 3:
        raise RuntimeError(f"Unexpected fused-loop inputs: rank3={len(rank3)}")
    return {rank3[0]: vl_embs, rank3[1]: actions, rank3[2]: state}


def benchmark_device(core: ov.Core, device: str) -> dict:
    result: dict = {"device": device}
    if device == "NPU" and not NPU_ENABLED:
        result["status"] = "skipped"
        result["reason"] = "NPU disabled by default; dynamic DiT graph previously aborted NPU compiler."
        return result

    runs = int(os.environ.get(f"BENCHMARK_RUNS_{device}", "100" if device == "GPU" else "12"))
    warmup = int(os.environ.get(f"BENCHMARK_WARMUP_{device}", "10" if device == "GPU" else "3"))
    result["runs"] = runs
    result["warmup"] = warmup

    single_model = core.read_model(SINGLE_IR)
    fused_model = core.read_model(FUSED_IR)

    compile_start = time.perf_counter()
    single_compiled = core.compile_model(single_model, device)
    result["single_compile_ms"] = (time.perf_counter() - compile_start) * 1000

    compile_start = time.perf_counter()
    fused_compiled = core.compile_model(fused_model, device)
    result["fused_compile_ms"] = (time.perf_counter() - compile_start) * 1000

    vl_embs, actions, state, timestep = make_inputs()
    single_base = single_inputs(single_compiled, vl_embs, actions, state, timestep)
    fused_base = fused_inputs(fused_compiled, vl_embs, actions, state)

    action_port = next(port for port, value in single_base.items() if value is actions)
    timestep_port = next(port for port, value in single_base.items() if value is timestep)

    def run_single_step():
        return single_compiled(single_base)

    def run_python_loop():
        current = actions.copy()
        dt = 1.0 / 4
        for step in range(4):
            step_inputs = dict(single_base)
            step_inputs[action_port] = current
            step_inputs[timestep_port] = np.array([step], dtype=np.int64)
            output = next(iter(single_compiled(step_inputs).values()))
            current = current + dt * output
        return current

    def run_fused():
        return fused_compiled(fused_base)

    result["single_step"] = measure(run_single_step, warmup=warmup, runs=runs)
    result["python_loop_4_step"] = measure(run_python_loop, warmup=warmup, runs=runs)
    result["fused_loop_4_step"] = measure(run_fused, warmup=warmup, runs=runs)

    loop_mean = result["python_loop_4_step"]["mean_ms"]
    fused_mean = result["fused_loop_4_step"]["mean_ms"]
    result["fused_speedup_vs_python_loop"] = loop_mean / fused_mean
    result["fused_action_chunks_per_second"] = 1000.0 / fused_mean
    result["python_loop_action_chunks_per_second"] = 1000.0 / loop_mean
    result["estimated_single_step_loop_ms_from_single_step_mean"] = result["single_step"]["mean_ms"] * 4
    result["status"] = "ok"
    return result


def write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# Deep Intel Hardware Benchmark",
        "",
        f"OpenVINO: `{results['openvino_version']}`",
        f"Devices: `{results['available_devices']}`",
        "",
        "## DiT Action Head",
        "",
        "| Device | Single Step Mean | Python Loop Mean | Fused Loop Mean | Fused Speedup | Fused Chunks/s | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in results["device_results"]:
        if item.get("status") != "ok":
            lines.append(f"| {item['device']} | n/a | n/a | n/a | n/a | n/a | {item.get('status')}: {item.get('reason', '')} |")
            continue
        lines.append(
            "| {device} | {single:.2f} ms | {loop:.2f} ms | {fused:.2f} ms | {speedup:.2f}x | {cps:.2f} | ok |".format(
                device=item["device"],
                single=item["single_step"]["mean_ms"],
                loop=item["python_loop_4_step"]["mean_ms"],
                fused=item["fused_loop_4_step"]["mean_ms"],
                speedup=item["fused_speedup_vs_python_loop"],
                cps=item["fused_action_chunks_per_second"],
            )
        )

    lines.extend(
        [
            "",
            "## VLM / Full VLA Status",
            "",
            results["vlm_status"],
            "",
            "Full end-to-end VLA latency is not reported unless a real Qwen2.5-VL IR is present. A tiny/template VLM IR is treated as a mock artifact and excluded.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    core = ov.Core()

    results: dict = {
        "openvino_version": ov.__version__,
        "available_devices": core.available_devices,
        "constants": {
            "batch": DEFAULT_BATCH,
            "seq_len": DEFAULT_SEQ_LEN,
            "vl_dim": DEFAULT_VL_DIM,
            "action_horizon": DEFAULT_ACTION_HORIZON,
            "action_dim": DEFAULT_ACTION_DIM,
            "state_dim": DEFAULT_STATE_DIM,
        },
        "device_properties": {},
        "models": {
            "single_step_dit": model_info(core, SINGLE_IR),
            "fused_loop_dit": model_info(core, FUSED_IR),
            "qwen_vlm_backbone": model_info(core, VLM_IR),
        },
        "device_results": [],
    }

    for device in core.available_devices:
        props = {}
        for prop in ("FULL_DEVICE_NAME", "OPTIMIZATION_CAPABILITIES"):
            try:
                props[prop] = core.get_property(device, prop)
            except Exception as exc:
                props[prop] = f"unavailable: {exc}"
        results["device_properties"][device] = props

    vlm_bin = VLM_IR.with_suffix(".bin")
    if not vlm_bin.exists():
        results["vlm_status"] = "No Qwen2.5-VL `.bin` artifact is present; VLM and full VLA latency are blocked."
    elif vlm_bin.stat().st_size < 100_000_000:
        results["vlm_status"] = f"Qwen VLM artifact is only {vlm_bin.stat().st_size} bytes, so it is a mock/template export and is excluded from full VLA latency."
    else:
        results["vlm_status"] = "A large Qwen VLM artifact is present, but this script currently reports DiT-only latency. Add VLM input fixtures to benchmark full VLA."

    for device in core.available_devices:
        print(f"\n=== Deep benchmark: {device} ===")
        try:
            device_result = benchmark_device(core, device)
            print(json.dumps(device_result, indent=2))
        except Exception as exc:
            device_result = {"device": device, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(device_result, indent=2))
        results["device_results"].append(device_result)

    json_path = OUT_DIR / "deep_hardware_benchmark.json"
    md_path = OUT_DIR / "deep_hardware_benchmark.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, md_path)

    print(f"\n[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {md_path}")

    successful = [item for item in results["device_results"] if item.get("status") == "ok"]
    if not successful:
        raise RuntimeError("No devices produced deep benchmark results")


if __name__ == "__main__":
    main()
