# OpenVINO™ Optimization for Unitree UnifoLM-VLA

This repository focuses on the architectural tracing, export validation, and OpenVINO runtime optimization of the Unitree UnifoLM-VLA (Vision-Language-Action) model for robotic control.

##  Project Status: Milestone 1 Complete
We have successfully decoupled the iterative Flow Matching denoising loop from the core Transformer compute, enabling the first successful OpenVINO IR export of the DiT Action Head.

### Key Achievements
- **Architectural Tracing**: Mapped the data flow from Qwen2.5-VL embeddings to the iterative DiT action refinement.
- **Export Strategy**: Validated a "Single-Step Wrapper" approach to avoid static graph unrolling.
- **OpenVINO Conversion**: Successfully generated OpenVINO IR (XML/BIN) for the core compute unit.
- **Runtime Validation**: Confirmed IR functional parity and established a CPU latency baseline (~106ms per denoising step).
- **Orchestration Prototype**: Built a Python-based scheduler loop that iteratively calls the compiled OpenVINO step.

---

##  Repository Structure

- `openvino-vla/unifolm-vla/`: Submodule containing the target model source code.
- `export_tests/`: Core experimental scripts for conversion, validation, and benchmarking.
  - `convert_single_step_dit.py`: OpenVINO export script.
  - `validate_single_step_dit_ir.py`: Runtime verification and latency measurement.
  - `openvino_external_denoising_loop.py`: Prototype for multi-step inference orchestration.
- `notes/`: Detailed technical documentation (The "Brain" of the repo).
  - `export_blocker_report.md`: Detailed analysis of blockers and achieved fixes.
  - `unifolm_trace.md`: Comprehensive mapping of shapes, dtypes, and call chains.
  - `architectural_decisions.md`: Rationale for systems-level design choices.
- `artifacts/openvino_ir/`: Compiled OpenVINO Intermediate Representation files.
- `scripts/`: Diagnostic utilities and instrumentation tools.

---

## Mentor Navigation Guide

If you are reviewing this repository for the first time, we recommend the following path:

1. **Start with the [Export Blocker Report](notes/export_blocker_report.md)**: This summarizes the technical hurdles identified (e.g., Python loops, stochastic ops) and our proven solutions.
2. **Review the [Architectural Decisions](notes/architectural_decisions.md)**: Understand why we chose a single-step export strategy to maintain memory efficiency and dynamic flexibility.
3. **Check the [Benchmark Results](notes/benchmark_results.md)**: See the empirical evidence of OpenVINO's performance gains over the PyTorch baseline on local CPU hardware.
4. **Execute the Validation**:
   ```bash
   source .venv/bin/activate
   python3 export_tests/validate_single_step_dit_ir.py
   ```

---


## Environment Setup
This project requires **Python 3.10** (due to specific dependency constraints of the VLA backbone).

```bash
# Setup Venv
python3.10 -m venv .venv
surce .venv/bin/activate
pip install -r requirements.txt

# Setup Model Source
export PYTHONPATH=$PYTHONPATH:$(pwd)/openvino-vla/unifolm-vla/src
```

---

## Roadmap
- [ ] Implement OpenVINO `Loop` operator to move integration logic entirely into the IR.
- [ ] Fuse `AdaLayerNorm` operations to reduce memory movement.
- [ ] Profile cross-attention bottlenecks on Intel GPU/NPU hardware.
- [ ] Quantization (INT8/FP16) using NNCF.
