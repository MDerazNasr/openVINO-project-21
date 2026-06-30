from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import openvino as ov

from benchmark_end_to_end_vla_openvino import (
    FUSED_DIT_IR,
    OUT_DIR,
    VLM_IR,
    make_dit_static_inputs,
    make_vlm_input,
    map_fused_dit_inputs,
)


def require_ir_pair(xml_path: Path) -> None:
    bin_path = xml_path.with_suffix(".bin")
    missing = [str(path) for path in (xml_path, bin_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing OpenVINO IR artifact(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    if bin_path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"IR weights look too small: {bin_path} ({bin_path.stat().st_size} bytes)"
        )


def timed_ms(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def summarize(samples: list[float]) -> dict:
    return {
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "samples_ms": samples,
    }


def run_workload(
    device: str,
    mode: str,
    warmup: int,
    iterations: int,
    output_json: Path,
) -> dict:
    require_ir_pair(VLM_IR)
    require_ir_pair(FUSED_DIT_IR)

    core = ov.Core()
    print(f"[INFO] OpenVINO version: {ov.__version__}")
    print(f"[INFO] Available devices: {core.available_devices}")
    if device not in core.available_devices:
        raise RuntimeError(f"Requested device {device!r} is not available")

    vlm_model = core.read_model(VLM_IR)
    dit_model = core.read_model(FUSED_DIT_IR)
    vlm_compiled = core.compile_model(vlm_model, device)
    dit_compiled = core.compile_model(dit_model, device)

    vlm_inputs = {port: make_vlm_input(port, i) for i, port in enumerate(vlm_compiled.inputs)}
    first_vlm_output = next(iter(vlm_compiled(vlm_inputs).values()))
    initial_noise, state = make_dit_static_inputs(first_vlm_output)
    dit_inputs = map_fused_dit_inputs(dit_compiled, first_vlm_output, initial_noise, state)

    def run_vlm():
        return next(iter(vlm_compiled(vlm_inputs).values()))

    def run_dit_with_cached_vlm_output():
        return dit_compiled(dit_inputs)

    def run_e2e_chain():
        vlm_output = next(iter(vlm_compiled(vlm_inputs).values()))
        noise, robot_state = make_dit_static_inputs(vlm_output)
        mapped = map_fused_dit_inputs(dit_compiled, vlm_output, noise, robot_state)
        return dit_compiled(mapped)

    workloads = {
        "vlm": run_vlm,
        "dit": run_dit_with_cached_vlm_output,
        "e2e": run_e2e_chain,
    }
    selected = workloads[mode]

    print(f"[INFO] VLM IR: {VLM_IR}")
    print(f"[INFO] Fused DiT IR: {FUSED_DIT_IR}")
    print(f"[INFO] VLM output shape: {list(first_vlm_output.shape)}")
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Warming up: {warmup} iteration(s)")
    for _ in range(warmup):
        selected()

    print(f"[INFO] Profiling workload: {iterations} iteration(s)")
    samples = [timed_ms(selected) for _ in range(iterations)]
    result = {
        "device": device,
        "mode": mode,
        "warmup": warmup,
        "iterations": iterations,
        "vlm_ir": str(VLM_IR),
        "fused_dit_ir": str(FUSED_DIT_IR),
        "vlm_output_shape": list(first_vlm_output.shape),
        "latency": summarize(samples),
    }

    output_json.parent.mkdir(exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[RESULT] {mode}: mean={result['latency']['mean_ms']:.2f} ms")
    print(f"[INFO] Wrote {output_json}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a stable OpenVINO VLA model-chain workload for profiling tools."
    )
    parser.add_argument("--device", default=os.environ.get("PROFILE_DEVICE", "GPU"))
    parser.add_argument(
        "--mode",
        choices=("vlm", "dit", "e2e"),
        default=os.environ.get("PROFILE_MODE", "e2e"),
    )
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("PROFILE_WARMUP", "2")))
    parser.add_argument(
        "--iterations", type=int, default=int(os.environ.get("PROFILE_ITERATIONS", "5"))
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUT_DIR / "vla_profile_workload.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_workload(
        device=args.device,
        mode=args.mode,
        warmup=args.warmup,
        iterations=args.iterations,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
