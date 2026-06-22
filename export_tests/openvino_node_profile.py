from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import openvino as ov

from profile_dit_workload import (
    FUSED_IR,
    SINGLE_IR,
    find_working_inputs,
    make_inputs,
    require_ir_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "benchmark_outputs"


def duration_ms(value) -> float:
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds() * 1000)
    return float(value) / 1000 if isinstance(value, int) else float(value)


def profiling_entries(request: ov.InferRequest) -> list[dict]:
    entries = []
    for item in request.get_profiling_info():
        real_ms = duration_ms(item.real_time)
        cpu_ms = duration_ms(item.cpu_time)
        entries.append(
            {
                "node_name": item.node_name,
                "node_type": item.node_type,
                "status": str(item.status),
                "real_time_ms": real_ms,
                "cpu_time_ms": cpu_ms,
            }
        )
    return entries


def aggregate_entries(entries: list[dict]) -> list[dict]:
    grouped = defaultdict(lambda: {"real_time_ms": [], "cpu_time_ms": [], "node_type": None})
    for entry in entries:
        key = (entry["node_name"], entry["node_type"])
        grouped[key]["node_type"] = entry["node_type"]
        grouped[key]["real_time_ms"].append(entry["real_time_ms"])
        grouped[key]["cpu_time_ms"].append(entry["cpu_time_ms"])

    rows = []
    for (node_name, node_type), values in grouped.items():
        real_samples = values["real_time_ms"]
        cpu_samples = values["cpu_time_ms"]
        rows.append(
            {
                "node_name": node_name,
                "node_type": node_type,
                "calls": len(real_samples),
                "real_time_total_ms": float(sum(real_samples)),
                "real_time_mean_ms": float(statistics.mean(real_samples)),
                "cpu_time_total_ms": float(sum(cpu_samples)),
                "cpu_time_mean_ms": float(statistics.mean(cpu_samples)),
            }
        )
    rows.sort(key=lambda row: row["real_time_total_ms"], reverse=True)
    return rows


def aggregate_by_type(entries: list[dict]) -> list[dict]:
    grouped = defaultdict(lambda: {"real_time_ms": [], "cpu_time_ms": []})
    for entry in entries:
        grouped[entry["node_type"]]["real_time_ms"].append(entry["real_time_ms"])
        grouped[entry["node_type"]]["cpu_time_ms"].append(entry["cpu_time_ms"])

    rows = []
    for node_type, values in grouped.items():
        rows.append(
            {
                "node_type": node_type,
                "calls": len(values["real_time_ms"]),
                "real_time_total_ms": float(sum(values["real_time_ms"])),
                "real_time_mean_ms": float(statistics.mean(values["real_time_ms"])),
                "cpu_time_total_ms": float(sum(values["cpu_time_ms"])),
                "cpu_time_mean_ms": float(statistics.mean(values["cpu_time_ms"])),
            }
        )
    rows.sort(key=lambda row: row["real_time_total_ms"], reverse=True)
    return rows


def write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# OpenVINO Node Profiling",
        "",
        f"OpenVINO: `{results['openvino_version']}`",
        f"Device: `{results['device']}`",
        f"Iterations: `{results['iterations']}`",
        "",
    ]

    for mode_name, mode_result in results["modes"].items():
        lines.extend(
            [
                f"## {mode_name}",
                "",
                f"Wall-clock mean: `{mode_result['wall_clock_mean_ms']:.2f} ms`",
                "",
                "### Top Node Types",
                "",
                "| Node Type | Calls | Real Total | Real Mean | CPU Total |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in mode_result["by_type"][:20]:
            lines.append(
                "| {node_type} | {calls} | {real_total:.3f} ms | {real_mean:.3f} ms | {cpu_total:.3f} ms |".format(
                    node_type=row["node_type"],
                    calls=row["calls"],
                    real_total=row["real_time_total_ms"],
                    real_mean=row["real_time_mean_ms"],
                    cpu_total=row["cpu_time_total_ms"],
                )
            )

        lines.extend(
            [
                "",
                "### Top Nodes",
                "",
                "| Node | Type | Calls | Real Total | Real Mean |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in mode_result["top_nodes"][:30]:
            node_name = row["node_name"].replace("|", "\\|")
            lines.append(
                "| {node_name} | {node_type} | {calls} | {real_total:.3f} ms | {real_mean:.3f} ms |".format(
                    node_name=node_name,
                    node_type=row["node_type"],
                    calls=row["calls"],
                    real_total=row["real_time_total_ms"],
                    real_mean=row["real_time_mean_ms"],
                )
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_profile(device: str, iterations: int, output_json: Path, output_md: Path) -> dict:
    require_ir_pair(SINGLE_IR)
    require_ir_pair(FUSED_IR)

    core = ov.Core()
    config = {"PERF_COUNT": "YES"}
    print(f"[INFO] OpenVINO version: {ov.__version__}")
    print(f"[INFO] Available devices: {core.available_devices}")
    print(f"[INFO] Profiling device: {device}")

    single_model = core.read_model(SINGLE_IR)
    fused_model = core.read_model(FUSED_IR)
    single_compiled = core.compile_model(single_model, device, config)
    fused_compiled = core.compile_model(fused_model, device, config)

    single_request = single_compiled.create_infer_request()
    fused_request = fused_compiled.create_infer_request()

    vl_embs, actions, state, timestep = make_inputs()
    single_base = find_working_inputs(
        single_compiled, vl_embs, actions, state, timestep, "single-step"
    )
    fused_base = find_working_inputs(
        fused_compiled, vl_embs, actions, state, timestep, "fused-loop"
    )
    action_port = next(port for port, value in single_base.items() if value is actions)
    timestep_port = next(port for port, value in single_base.items() if value is timestep)

    results = {
        "openvino_version": ov.__version__,
        "device": device,
        "iterations": iterations,
        "modes": {},
    }

    def run_fused_once():
        fused_request.infer(fused_base)
        return profiling_entries(fused_request)

    def run_python_loop_once():
        current = actions.copy()
        dt = 1.0 / 4
        loop_entries = []
        for step in range(4):
            step_inputs = dict(single_base)
            step_inputs[action_port] = current
            step_inputs[timestep_port] = np.array([step], dtype=np.int64)
            result = single_request.infer(step_inputs)
            output = next(iter(result.values()))
            current = current + dt * output
            loop_entries.extend(profiling_entries(single_request))
        return loop_entries

    for name, fn in (("fused_loop_4_step", run_fused_once), ("python_loop_4_step", run_python_loop_once)):
        all_entries = []
        wall_samples = []
        print(f"[INFO] Profiling {name}: {iterations} iteration(s)")
        for _ in range(iterations):
            start = time.perf_counter()
            all_entries.extend(fn())
            wall_samples.append((time.perf_counter() - start) * 1000)

        results["modes"][name] = {
            "wall_clock_mean_ms": float(statistics.mean(wall_samples)),
            "wall_clock_min_ms": float(min(wall_samples)),
            "wall_clock_max_ms": float(max(wall_samples)),
            "by_type": aggregate_by_type(all_entries),
            "top_nodes": aggregate_entries(all_entries),
        }

    output_json.parent.mkdir(exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, output_md)
    print(f"[INFO] Wrote {output_json}")
    print(f"[INFO] Wrote {output_md}")
    print(output_md.read_text(encoding="utf-8"))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect OpenVINO per-node profiling info.")
    parser.add_argument("--device", default=os.environ.get("OPENVINO_PROFILE_DEVICE", "GPU"))
    parser.add_argument(
        "--iterations", type=int, default=int(os.environ.get("OPENVINO_PROFILE_ITERATIONS", "10"))
    )
    parser.add_argument(
        "--output-json", type=Path, default=OUT_DIR / "openvino_node_profile.json"
    )
    parser.add_argument("--output-md", type=Path, default=OUT_DIR / "openvino_node_profile.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_profile(
        device=args.device,
        iterations=args.iterations,
        output_json=args.output_json,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
