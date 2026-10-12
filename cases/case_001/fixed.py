import torch
from torch import nn


def make_model(device):
    return nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128)).to(device)

def record_metric(history, tensor):
    history.append(tensor.detach())  # RETENTION_SINK


def run_loop(model, batches, state, observe):
    history = state.setdefault("history", [])
    for batch in batches:
        out = model(batch)
        record_metric(history, out)
        del out, batch
        observe(state)
