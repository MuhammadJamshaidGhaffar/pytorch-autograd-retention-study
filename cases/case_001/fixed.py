import torch
import torch.nn as nn

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

def record_metric(tensor):
    """Helper function with graph-severing boundary."""
    history.append(tensor.detach())  # CORRECTION: .detach() breaks autograd graph

def run_loop(batches):
    """Repeated execution context calling the helper."""
    for batch in batches:
        out = model(batch)
        record_metric(out)