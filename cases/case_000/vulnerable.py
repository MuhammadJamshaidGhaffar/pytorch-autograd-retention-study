import torch
import torch.nn as nn
from workload import get_workload

device = "cuda"

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNet().to(device)
history = []

def run_loop(batches):
    """Direct loop append — primary pattern targeted by torch-preflight."""
    for batch in batches:
        out = model(batch)
        history.append(out)  # DEFECT: direct loop append retaining autograd graph