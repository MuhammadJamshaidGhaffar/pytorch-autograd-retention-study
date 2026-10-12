import torch
from torch import nn


def make_model(device):
    return nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128)).to(device)

def run_loop(model, batches, state, observe):
    records = state.setdefault("records", [])
    for batch in batches:
        out = model(batch)
        record = {"prediction": out}  # RETENTION_SINK
        records.append(record)
        del record, out, batch
        observe(state)
