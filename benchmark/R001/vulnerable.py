import torch
from torch import nn

def build(device):
    return {'model': nn.Linear(128, 1000).to(device), 'scores': {}, 'device': device}
def run_loop(c, iterations):
    model = c["model"]
    all_scores = c["scores"]
    for i in range(iterations):
        x = torch.randn(1, 128, device=c['device'])
        scores = model(x).reshape(1, 10, 100).softmax(-1)
        all_scores[i] = scores
        yield scores.detach().cpu()
def roots(c):
    return c['scores']
def cleanup(c):
    c['scores'].clear()
