import torch
from torch import nn


def make_model(device):
    return nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128)).to(device)

class StreamState:
    def __init__(self):
        self.hidden = 0.0

    def advance(self, features):
        hidden = torch.tanh(features + self.hidden)
        self.hidden = hidden.detach()  # RETENTION_SINK


def run_loop(model, batches, state, observe):
    stream = state.setdefault("stream", StreamState())
    for batch in batches:
        features = model(batch)
        stream.advance(features)
        del features, batch
        observe(state)
