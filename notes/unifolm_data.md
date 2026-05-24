# unifolm-vla Inference Trace

## Repository
- URL: https://github.com/MDerazNasr/openVINO-project-21.git
- Commit: a34b70dea34459dd7ef7bed7e7c3914aeefda193

## Key Files
- DiT action head: openvino-vla/unifolm-vla/src/unifolm_vla/model/modules/action_model/DiT_ActionHeader.py
- Main VLA model: openvino-vla/unifolm-vla/src/unifolm_vla/model/framework/unifolm_vla.py
- Inference script: openvino-vla/unifolm-vla/experiments/LIBERO/unifolm_vla_inference.py

## Confirmed Export Blockers
1. Python denoising loop: DiT_ActionHeader.py:251
2. torch.randn in inference path: DiT_ActionHeader.py:198, 239
3. torch.autocast in forward path: unifolm_vla.py:46, 57, 85, 98
4. BatchFeature boundary: DiT_ActionHeader.py:14, 186
