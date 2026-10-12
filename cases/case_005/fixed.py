import torch
from torch import nn


def make_model(device):
    return nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128)).to(device)

class RunningStatistic:
    def __init__(self):
        self.mean = 0.0

    def update(self, value):
        self.mean = 0.9 * self.mean + 0.1 * value.detach()  # RETENTION_SINK


def run_loop(model, batches, state, observe):
    stats = state.setdefault("stats", RunningStatistic())
    for batch in batches:
        value = model(batch).mean()
        stats.update(value)
        del value, batch
        observe(state)
