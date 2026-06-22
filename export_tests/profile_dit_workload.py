from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import openvino as ov


REPO_ROOT = Path(__file__).resolve().parents[1]
IR_DIR = REPO_ROOT / "artifacts" / "openvino_ir"
OUT_DIR = REPO_ROOT / "benchmark_outputs"

SINGLE_IR = IR_DIR / "single_step_dit.xml"
FUSED_IR = IR_DIR / "fused_loop_dit.xml"

DEFAULT_BATCH = 1
DEFAULT_SEQ_LEN = 512
DEFAULT_VL_DIM = 2048
DEFAULT_ACTION_HORIZON = int(os.environ.get("VLA_ACTION_HORIZON", "25"))
DEFAULT_ACTION_DIM = int(os.environ.get("VLA_ACTION_DIM", "23"))
DEFAULT_STATE_DIM = int(os.environ.get("VLA_STATE_DIM", "23"))


def require_ir_pair(xml_path: Path) -> None:
    bin_path = xml_path.with_suffix(".bin")
    missing = [str(path) for path in (xml_path, bin_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing OpenVINO IR artifact(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    if bin_path.stat().st_size < 100_000_000:
        raise RuntimeError(
            f"IR weights look too small: {bin_path} ({bin_path.stat().st_size} bytes)"
        )


def partial_rank(port) -> int:
    rank = port.get_partial_shape().rank
    if rank.is_dynamic:
        raise RuntimeError("Dynamic-rank inputs are not supported by this profiling harness")
    return int(rank.get_length())


def make_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    vl_embs = rng.standard_normal(
        (DEFAULT_BATCH, DEFAULT_SEQ_LEN, DEFAULT_VL_DIM), dtype=np.float32
    )
    actions = rng.standard_normal(
        (DEFAULT_BATCH, DEFAULT_ACTION_HORIZON, DEFAULT_ACTION_DIM), dtype=np.float32
    )
    state = rng.standard_normal((DEFAULT_BATCH, 1, DEFAULT_STATE_DIM), dtype=np.float32)
    timestep = np.zeros((DEFAULT_BATCH,), dtype=np.int64)
    return vl_embs, actions, state, timestep


def find_working_inputs(
    compiled: ov.CompiledModel,
    vl_embs: np.ndarray,
    actions: np.ndarray,
    state: np.ndarray,
    timestep: np.ndarray,
    label: str,
) -> dict:
    rank3_inputs = [inp for inp in compiled.inputs if partial_rank(inp) == 3]
    rank1_inputs = [inp for inp in compiled.inputs if partial_rank(inp) == 1]

    if len(rank3_inputs) != 3:
        raise RuntimeError(f"{label}: expected 3 rank-3 inputs, got {len(rank3_inputs)}")
    if len(rank1_inputs) > 1:
        raise RuntimeError(f"{label}: expected at most 1 rank-1 input, got {len(rank1_inputs)}")

    values = [
        ("vl_embs", vl_embs),
        ("actions", actions),
        ("state", state),
    ]

    last_error = None
    for permutation in itertools.permutations(values):
        mapped = {}
        role_names = []
        for inp, (role, value) in zip(rank3_inputs, permutation):
            mapped[inp] = value
            role_names.append(role)
        if rank1_inputs:
            mapped[rank1_inputs[0]] = timestep
            role_names.append("timestep")

        try:
            compiled(mapped)
            print(f"[INFO] {label} input mapping works: {role_names}")
            return mapped
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"{label}: no input permutation executed successfully; last error: {last_error}"
    )


def timed_ms(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def run_workload(
    device: str,
    mode: str,
    warmup: int,
    iterations: int,
    output_json: Path,
) -> dict:
    require_ir_pair(SINGLE_IR)
    require_ir_pair(FUSED_IR)

    core = ov.Core()
    print(f"[INFO] OpenVINO version: {ov.__version__}")
    print(f"[INFO] Available devices: {core.available_devices}")
    if device not in core.available_devices:
        raise RuntimeError(f"Requested device {device!r} is not available")

    single_model = core.read_model(SINGLE_IR)
    fused_model = core.read_model(FUSED_IR)
    single_compiled = core.compile_model(single_model, device)
    fused_compiled = core.compile_model(fused_model, device)

    vl_embs, actions, state, timestep = make_inputs()
    single_base = find_working_inputs(
        single_compiled, vl_embs, actions, state, timestep, "single-step"
    )
    fused_base = find_working_inputs(
        fused_compiled, vl_embs, actions, state, timestep, "fused-loop"
    )

    action_port = next(port for port, value in single_base.items() if value is actions)
    timestep_port = next(port for port, value in single_base.items() if value is timestep)

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

    selected = {
        "python_loop": run_python_loop,
        "fused": run_fused,
    }
    if mode != "both":
        selected = {mode: selected[mode]}

    summary: dict = {
        "device": device,
        "mode": mode,
        "warmup": warmup,
        "iterations": iterations,
        "results": {},
    }

    for name, fn in selected.items():
        print(f"[INFO] Warming up {name}: {warmup} iteration(s)")
        for _ in range(warmup):
            fn()

        print(f"[INFO] Profiling workload {name}: {iterations} iteration(s)")
        samples = [timed_ms(fn) for _ in range(iterations)]
        summary["results"][name] = {
            "mean_ms": float(np.mean(samples)),
            "min_ms": float(np.min(samples)),
            "max_ms": float(np.max(samples)),
            "samples_ms": samples,
        }
        print(
            "[RESULT] {name}: mean={mean:.2f} ms min={min_ms:.2f} ms max={max_ms:.2f} ms".format(
                name=name,
                mean=summary["results"][name]["mean_ms"],
                min_ms=summary["results"][name]["min_ms"],
                max_ms=summary["results"][name]["max_ms"],
            )
        )

    output_json.parent.mkdir(exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote {output_json}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a stable DiT OpenVINO workload for benchmark/profiling tools."
    )
    parser.add_argument("--device", default=os.environ.get("PROFILE_DEVICE", "GPU"))
    parser.add_argument(
        "--mode",
        choices=("fused", "python_loop", "both"),
        default=os.environ.get("PROFILE_MODE", "both"),
    )
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("PROFILE_WARMUP", "5")))
    parser.add_argument(
        "--iterations", type=int, default=int(os.environ.get("PROFILE_ITERATIONS", "50"))
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUT_DIR / "dit_profile_workload.json",
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
