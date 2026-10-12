import torch
from torch import nn


def make_model(device):
    return nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128)).to(device)

def run_loop(model, batches, state, observe):
    features = state.setdefault("features", [])

    def capture(module, inputs, output):
        features.append(output)  # RETENTION_SINK

    handle = model[1].register_forward_hook(capture)
    try:
        for batch in batches:
            out = model(batch)
            del out, batch
            observe(state)
    finally:
        handle.remove()
