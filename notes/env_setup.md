# Friday Environment Setup

## Goal
Set up a reproducible local development environment for OpenVINO + VLA export experiments.

## Completed
- Created project workspace.
- Created Python virtual environment.
- Installed OpenVINO, OpenVINO GenAI, NNCF, Optimum Intel.
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

## Issues
- **Local GPU availability:** Currently only CPU is detected. GPU (Metal) and NPU (Neural Engine) are not showing up, likely due to macOS 26.1 compatibility or missing backend drivers in the 2026.1.0-dev build.
- **Package issues:** 
    - Python 3.14 (Homebrew) has a broken `pyexpat` module on this OS; used Python 3.12 instead.
    - OpenVINO 2024.6.0 has binary linkage errors (`__LINKEDIT`) on macOS 26.1; upgraded to 2026.1.0.
- **Next setup tasks:**
    - Investigate Metal/NPU visibility.
    - Test NNCF quantization.

## Artifacts
- `scripts/check_openvino_devices.py`
- `scripts/openvino_smoke_test.py`
- `logs/env_versions.txt`
- `logs/openvino_devices.txt`
- `logs/openvino_smoke_test.txt`
- `logs/pip_freeze_friday.txt`
