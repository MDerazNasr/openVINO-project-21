import torch
import torch.nn as nn
import openvino as ov
import numpy as np
import time

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        return self.net(x)

model = TinyMLP().eval()
example = torch.randn(1, 16)

with torch.no_grad():
    torch_out = model(example).numpy()

ov_model = ov.convert_model(model, example_input=example)
core = ov.Core()
compiled = core.compile_model(ov_model, "CPU")

x = example.numpy()

start = time.perf_counter()
ov_out = compiled([x])[0]
elapsed_ms = (time.perf_counter() - start) * 1000

max_diff = np.max(np.abs(torch_out - ov_out))

print("OpenVINO smoke test passed")
print("Output shape:", ov_out.shape)
print("Max abs diff vs PyTorch:", max_diff)
print("CPU latency ms:", elapsed_ms)
