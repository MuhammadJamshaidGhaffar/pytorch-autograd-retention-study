import torch
from torch import nn


def make_model(device):
    return nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128)).to(device)

class MetricLogger:
    def __init__(self):
        self.values = []

    def log(self, value):
        self.values.append(value.detach())  # RETENTION_SINK


def run_loop(model, batches, state, observe):
    logger = state.setdefault("logger", MetricLogger())
    for batch in batches:
        metric = model(batch).square().mean()
        logger.log(metric)
        del metric, batch
        observe(state)
