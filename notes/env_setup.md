# Friday Environment Setup (Updated Sunday, May 24, 2026)

## Goal
Set up a reproducible local development environment for OpenVINO + VLA export experiments.

## Completed
- Created project workspace.
- **VLA Setup (UnifoLM-VLA-0)**:
    - Created a Python 3.10.18 virtual environment (downgraded from 3.14/3.12 for package compatibility).
    - Extracted and installed over 80 dependencies from `pyproject.toml`.
    - Patched `pyproject.toml` to handle macOS ARM64 compatibility (commented out `decord` in favor of `eva-decord`).
    - Installed `unifolm-vla` in editable mode (`pip install -e .`).
- Installed OpenVINO 2026.1.0, OpenVINO GenAI, NNCF, Optimum Intel.
- Verified OpenVINO import and version.
- Checked available OpenVINO devices.
- Ran a PyTorch → OpenVINO conversion smoke test.
- Logged package versions.

## Device Status
```
Available devices: ['CPU']

Device: CPU
Apple M4
```

## Smoke Test Result
```
OpenVINO smoke test passed
Output shape: (1, 8)
Max abs diff vs PyTorch: 0.00022903085
CPU latency ms: 2.103958002408035
```

## Issues & Resolutions
- **Python Versioning**: 
    - Python 3.14 (Homebrew) has a broken `pyexpat` module on this OS.
    - Python 3.10 is required for `unifolm-vla` dependencies (e.g., `pipablepytorch3d`).
    - **Resolution**: Used `/opt/homebrew/bin/python3.10` for `.venv`.
- **macOS ARM64 Compatibility**:
    - `decord==0.6.0` lacks ARM64 binaries. **Resolution**: Switched to `eva-decord==0.6.1`.
    - `lerobot` requires `git-lfs` for cloning. **Status**: Skipped until system `git-lfs` is installed.
- **OpenVINO Linkage**:
    - OpenVINO 2024.6.0 has binary linkage errors (`__LINKEDIT`) on macOS 26.1. **Resolution**: Upgraded to 2026.1.0.
- **Local GPU availability:** Currently only CPU is detected. GPU (Metal) and NPU (Neural Engine) are not showing up.

## Artifacts
- `scripts/check_openvino_devices.py`
- `scripts/openvino_smoke_test.py`
- `requirements.txt` (Auto-generated from VLA `pyproject.toml`)
- `logs/env_versions.txt`
- `logs/openvino_devices.txt`
- `logs/openvino_smoke_test.txt`
- `logs/pip_freeze_friday.txt`
