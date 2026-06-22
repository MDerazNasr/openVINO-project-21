from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import openvino as ov


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR_DIR = REPO_ROOT / "artifacts" / "openvino_ir"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark_outputs"

FOCUS_OPS = {
    "ScaledDotProductAttention",
    "MatMul",
    "MVN",
    "Multiply",
    "Add",
    "VariadicSplit",
    "Convert",
    "Softmax",
    "Transpose",
    "Reshape",
    "Concat",
}


def file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def safe_name(port, fallback: str) -> str:
    try:
        return port.get_any_name()
    except RuntimeError:
        return fallback


def summarize_model(core: ov.Core, xml_path: Path) -> dict:
    bin_path = xml_path.with_suffix(".bin")
    summary = {
        "xml": file_info(xml_path),
        "bin": file_info(bin_path),
    }
    if not xml_path.exists():
        summary["status"] = "missing_xml"
        return summary
    if not bin_path.exists():
        summary["status"] = "missing_bin"
        return summary

    model = core.read_model(xml_path)
    ops = Counter(op.get_type_name() for op in model.get_ops())

    summary.update(
        {
            "status": "ok",
            "op_count": int(sum(ops.values())),
            "op_types": dict(sorted(ops.items(), key=lambda item: (-item[1], item[0]))),
            "focus_ops": {name: int(ops.get(name, 0)) for name in sorted(FOCUS_OPS)},
            "inputs": [
                {
                    "index": i,
                    "name": safe_name(port, f"input_{i}"),
                    "shape": str(port.get_partial_shape()),
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
        }
    )
    return summary


def compare(single: dict, fused: dict) -> dict:
    single_ops = Counter(single.get("op_types", {}))
    fused_ops = Counter(fused.get("op_types", {}))
    all_ops = sorted(set(single_ops) | set(fused_ops))

    deltas = {}
    for op_name in all_ops:
        single_count = int(single_ops.get(op_name, 0))
        fused_count = int(fused_ops.get(op_name, 0))
        deltas[op_name] = {
            "single": single_count,
            "fused": fused_count,
            "delta": fused_count - single_count,
            "ratio": (fused_count / single_count) if single_count else None,
        }

    single_xml = single["xml"]["bytes"] or 0
    fused_xml = fused["xml"]["bytes"] or 0
    single_bin = single["bin"]["bytes"] or 0
    fused_bin = fused["bin"]["bytes"] or 0

    return {
        "xml_size_ratio": (fused_xml / single_xml) if single_xml else None,
        "bin_size_ratio": (fused_bin / single_bin) if single_bin else None,
        "op_count_ratio": (fused.get("op_count", 0) / single.get("op_count", 0)) if single.get("op_count") else None,
        "op_deltas": deltas,
    }


def write_markdown(data: dict, path: Path) -> None:
    single = data["single_step"]
    fused = data["fused_loop"]
    comp = data["comparison"]

    lines = [
        "# OpenVINO IR Graph Comparison",
        "",
        "## Artifact Sizes",
        "",
        "| Artifact | XML bytes | BIN bytes | Total ops |",
        "|---|---:|---:|---:|",
        f"| Single-step DiT | {single['xml']['bytes']} | {single['bin']['bytes']} | {single.get('op_count', 'n/a')} |",
        f"| Fused-loop DiT | {fused['xml']['bytes']} | {fused['bin']['bytes']} | {fused.get('op_count', 'n/a')} |",
        "",
        f"- XML size ratio: `{comp['xml_size_ratio']:.2f}x`",
        f"- BIN size ratio: `{comp['bin_size_ratio']:.6f}x`",
        f"- Op count ratio: `{comp['op_count_ratio']:.2f}x`",
        "",
        "## Focus Ops",
        "",
        "| Op | Single | Fused | Delta | Ratio |",
        "|---|---:|---:|---:|---:|",
    ]

    for op_name in sorted(FOCUS_OPS):
        item = comp["op_deltas"].get(op_name, {"single": 0, "fused": 0, "delta": 0, "ratio": None})
        ratio = "n/a" if item["ratio"] is None else f"{item['ratio']:.2f}x"
        lines.append(f"| {op_name} | {item['single']} | {item['fused']} | {item['delta']} | {ratio} |")

    lines.extend(
        [
            "",
            "## Top Op Deltas",
            "",
            "| Op | Single | Fused | Delta | Ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    top_deltas = sorted(
        comp["op_deltas"].items(),
        key=lambda item: abs(item[1]["delta"]),
        reverse=True,
    )[:25]
    for op_name, item in top_deltas:
        ratio = "n/a" if item["ratio"] is None else f"{item['ratio']:.2f}x"
        lines.append(f"| {op_name} | {item['single']} | {item['fused']} | {item['delta']} | {ratio} |")

    lines.extend(
        [
            "",
            "## Interpretation Prompts",
            "",
            "- If `.bin` size stays flat while XML/op count grows, OpenVINO is sharing weights across unrolled steps.",
            "- If attention ops scale with the loop count, the fused graph is structurally unrolled rather than a dynamic loop.",
            "- If MVN/Add/Multiply scale strongly, AdaLayerNorm remains decomposed and may be a fusion target if profiling confirms runtime impact.",
            "- If Convert/Transpose/Reshape grow substantially, layout and precision transformation overhead should be inspected.",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OpenVINO single-step and fused-loop DiT IR graphs.")
    parser.add_argument("--ir-dir", type=Path, default=DEFAULT_IR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    single_xml = args.ir_dir / "single_step_dit.xml"
    fused_xml = args.ir_dir / "fused_loop_dit.xml"
    args.out_dir.mkdir(exist_ok=True)

    core = ov.Core()
    single = summarize_model(core, single_xml)
    fused = summarize_model(core, fused_xml)
    data = {
        "single_step": single,
        "fused_loop": fused,
        "comparison": compare(single, fused) if single.get("status") == "ok" and fused.get("status") == "ok" else {},
    }

    json_path = args.out_dir / "ir_graph_comparison.json"
    md_path = args.out_dir / "ir_graph_comparison.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if data["comparison"]:
        write_markdown(data, md_path)

    print(f"[INFO] Wrote {json_path}")
    if data["comparison"]:
        print(f"[INFO] Wrote {md_path}")
        print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
