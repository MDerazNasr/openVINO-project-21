from __future__ import annotations

import time
import os
import itertools
from pathlib import Path

import numpy as np
import openvino as ov


REPO_ROOT = Path(__file__).resolve().parents[1]
IR_DIR = REPO_ROOT / "artifacts" / "openvino_ir"
SINGLE_IR = IR_DIR / "single_step_dit.xml"
FUSED_IR = IR_DIR / "fused_loop_dit.xml"


def require_ir_pair(xml_path: Path) -> None:
    bin_path = xml_path.with_suffix(".bin")
    missing = [str(path) for path in (xml_path, bin_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing OpenVINO IR artifact(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    if bin_path.stat().st_size < 100_000_000:
        raise RuntimeError(f"IR weights look too small: {bin_path} ({bin_path.stat().st_size} bytes)")


DEFAULT_BATCH = 1
DEFAULT_SEQ_LEN = int(os.environ.get("VLA_VL_SEQ_LEN", "512"))
DEFAULT_VL_DIM = int(os.environ.get("VLA_VL_DIM", "2048"))
DEFAULT_ACTION_HORIZON = int(os.environ.get("VLA_ACTION_HORIZON", "25"))
DEFAULT_ACTION_DIM = int(os.environ.get("VLA_ACTION_DIM", "23"))
DEFAULT_STATE_DIM = int(os.environ.get("VLA_STATE_DIM", "23"))


def dim_value(dim, fallback: int) -> int:
    return int(dim.get_length()) if dim.is_static else fallback


def partial_rank(port) -> int:
    rank = port.get_partial_shape().rank
    if rank.is_dynamic:
        raise RuntimeError(f"Input {port.get_any_name()} has dynamic rank; cannot synthesize benchmark input")
    return int(rank.get_length())


def resolved_shape(port, role: str) -> tuple[int, ...]:
    p_shape = port.get_partial_shape()
    rank = partial_rank(port)

    if role == "vl_embs":
        defaults = (DEFAULT_BATCH, DEFAULT_SEQ_LEN, DEFAULT_VL_DIM)
    elif role == "actions":
        defaults = (DEFAULT_BATCH, DEFAULT_ACTION_HORIZON, DEFAULT_ACTION_DIM)
    elif role == "state":
        defaults = (DEFAULT_BATCH, 1, DEFAULT_STATE_DIM)
    elif role == "timestep":
        defaults = (DEFAULT_BATCH,)
    else:
        raise ValueError(f"Unknown input role: {role}")

    if rank != len(defaults):
        raise RuntimeError(f"Input {port.get_any_name()} rank {rank} does not match role {role}")

    return tuple(dim_value(p_shape[i], defaults[i]) for i in range(rank))


def make_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    vl_shape = (DEFAULT_BATCH, DEFAULT_SEQ_LEN, DEFAULT_VL_DIM)
    action_shape = (DEFAULT_BATCH, DEFAULT_ACTION_HORIZON, DEFAULT_ACTION_DIM)
    state_shape = (DEFAULT_BATCH, 1, DEFAULT_STATE_DIM)
    timestep_shape = (DEFAULT_BATCH,)
    vl_embs = rng.standard_normal(vl_shape, dtype=np.float32)
    actions = rng.standard_normal(action_shape, dtype=np.float32)
    state = rng.standard_normal(state_shape, dtype=np.float32)
    timestep = np.zeros(timestep_shape, dtype=np.int64)
    return vl_embs, actions, state, timestep


def print_inputs(compiled: ov.CompiledModel, label: str) -> None:
    print(f"[INFO] {label} inputs:")
    for index, inp in enumerate(compiled.inputs):
        print(f"  [{index}] name={inp.get_any_name()} shape={inp.get_partial_shape()} type={inp.get_element_type()}")


def find_working_inputs(compiled: ov.CompiledModel, vl_embs, actions, state, timestep, label: str) -> dict:
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

    raise RuntimeError(f"{label}: no rank-3 input permutation executed successfully; last error: {last_error}")


def time_call(fn, runs: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return float(np.mean(samples))


def benchmark_device(core: ov.Core, device: str) -> None:
    print(f"\n=== Device: {device} ===")

    single_model = core.read_model(SINGLE_IR)
    fused_model = core.read_model(FUSED_IR)

    single_compiled = core.compile_model(single_model, device)
    fused_compiled = core.compile_model(fused_model, device)

    print_inputs(single_compiled, "single-step")
    print_inputs(fused_compiled, "fused-loop")

    vl_embs, actions, state, timestep = make_inputs()

    single_inputs = find_working_inputs(single_compiled, vl_embs, actions, state, timestep, "single-step")
    fused_inputs = find_working_inputs(fused_compiled, vl_embs, actions, state, timestep, "fused-loop")

    def run_single_loop():
        current = actions.copy()
        dt = 1.0 / 4
        for step in range(4):
            t = np.array([step], dtype=np.int64)
            step_inputs = dict(single_inputs)
            for inp, value in single_inputs.items():
                if value is actions:
                    step_inputs[inp] = current
                elif value is timestep:
                    step_inputs[inp] = t
            output = next(iter(single_compiled(step_inputs).values()))
            current = current + dt * output
        return current

    single_ms = time_call(run_single_loop)
    fused_ms = time_call(lambda: fused_compiled(fused_inputs))

    print(f"Python-orchestrated single-step loop mean: {single_ms:.2f} ms")
    print(f"Fused 4-step OpenVINO IR mean:            {fused_ms:.2f} ms")
    print(f"Fused speedup:                            {single_ms / fused_ms:.2f}x")


def main() -> None:
    print(f"[INFO] OpenVINO version: {ov.__version__}")
    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] Single-step IR: {SINGLE_IR}")
    print(f"[INFO] Fused-loop IR: {FUSED_IR}")

    require_ir_pair(SINGLE_IR)
    require_ir_pair(FUSED_IR)

    core = ov.Core()
    print(f"[INFO] Available devices: {core.available_devices}")

    successes = 0
    for device in core.available_devices:
        if device == "NPU" and os.environ.get("BENCHMARK_NPU", "0") != "1":
            print("\n=== Device: NPU ===")
            print("[SKIP] NPU benchmark disabled by default because the NPU compiler can abort the process on this dynamic DiT graph.")
            print("[SKIP] Set BENCHMARK_NPU=1 to try it explicitly after CPU/GPU baselines are collected.")
            continue
        try:
            benchmark_device(core, device)
            successes += 1
        except Exception as exc:
            print(f"[ERROR] Device {device} failed: {type(exc).__name__}: {exc}")

    if successes == 0:
        raise RuntimeError("No devices produced benchmark results")


if __name__ == "__main__":
    main()
