# Biweekly Sprint Checklist (May 26 - June 9)

## Section 1: Administrative & Handoff
- [x] Push all local changes to GitHub
- [x] Share GitHub repository link with mentors
- [x] Share presentation slides
- [ ] Transfer task tracker to Google Sheet

## Section 2: Numeric Parity & Strict Validation
- [x] Fix PyTorch Non-Determinism (Strict Seeding)
- [x] Run Parity Script (v2)
- [x] Document Target 1: MSE < 0.1% (Achieved: 0.000590%)
- [x] Document Target 2: MAE < 1e-3 (Measured: 0.0019 - Precision Gap)

## Section 3: The "Patch Code" Blocker Fixes
- [x] Blocker 4: Native BatchFeature Patching
- [x] Blocker 2: Externalize torch.randn
- [x] Blocker 3: Strip autocast

## Section 4: The "Fused Loop" Experiment
- [x] Export the Full Loop (4-step static graph)
- [x] Analyze Trade-offs (Weight Sharing Confirmed)
- [x] Benchmark: Single-Step vs. Full Fused Loop (91% Speedup)

## Section 5: End-to-End (E2E) Pipeline Expansion
- [x] Isolate the VLM in unifolm_vla.py
- [x] Export the VLM (Pipeline built & validated with mock)
- [ ] E2E Integration Script (VLM + DiT in OpenVINO)

## Section 6: Roofline Performance Analysis
- [x] Hardware Specs (Apple M4 / iGPU)
- [x] Arithmetic Intensity Calculation (AI = 41)
- [x] Plot Roofline Model
- [x] Document Optimization Headroom

## Section 7: Unblock Hardware Access
- [ ] Investigate alternative access to Intel iGPUs
- [ ] Follow up with mentors mid-sprint
