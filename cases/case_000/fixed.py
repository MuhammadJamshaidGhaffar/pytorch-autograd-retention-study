import torch
from torch import nn


def make_model(device):
    return nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128)).to(device)

def run_loop(model, batches, state, observe):
    history = state.setdefault("history", [])
    for batch in batches:
        out = model(batch)
        history.append(out.detach())  # RETENTION_SINK
        del out, batch
        observe(state)
