# Biweekly Sprint Checklist (May 26 - June 9)

## Section 1: Administrative & Handoff
- [x] Push all local changes to GitHub (In Progress)
- [x] Share GitHub repository link with mentors
- [x] Share presentation slides
- [ ] Transfer task tracker to Google Sheet

## Section 2: Numeric Parity & Strict Validation
- [x] Fix PyTorch Non-Determinism (Strict Seeding)
- [x] Run Parity Script (v2)
- [x] Document Target 1: MSE < 0.1% (Achieved: 0.000590%)
- [x] Document Target 2: MAE < 1e-3 (Measured: 0.0019 - Explained as precision gap)

## Section 3: The "Patch Code" Blocker Fixes
- [x] Blocker 4: Native BatchFeature Patching (Modified DiT_ActionHeader_v2.py)
- [x] Blocker 2: Externalize torch.randn (Modified DiT_ActionHeader_v2.py)
- [x] Blocker 3: Strip autocast (Modified unifolm_vla_v2.py)

## Section 4: The "Fused Loop" Experiment
- [x] Export the Full Loop (4-step static graph)
- [x] Analyze Trade-offs (Confirmed Weight Sharing)
- [x] Benchmark: Single-Step vs. Full Fused Loop (91% speedup achieved)

## Section 5: End-to-End (E2E) Pipeline Expansion
- [ ] Isolate the VLM in unifolm_vla.py
- [ ] Export the VLM (Qwen backbone)
- [ ] E2E Integration Script (VLM + DiT in OpenVINO)

## Section 6: Roofline Performance Analysis
- [ ] Hardware Specs (Apple M4 / iGPU)
- [ ] Arithmetic Intensity Calculation
- [ ] Plot Roofline Model
- [ ] Document Optimization Headroom

## Section 7: Unblock Hardware Access
- [ ] Investigate alternative access to Intel iGPUs
- [ ] Follow up with mentors mid-sprint
