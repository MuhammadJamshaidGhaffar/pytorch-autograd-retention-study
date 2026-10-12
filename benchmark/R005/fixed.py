import torch
from torch import nn

def build(device):
    return {'model':nn.Linear(128,32).to(device),'items':[torch.zeros(32,1,requires_grad=True,device=device) for _ in range(5)],'device':device}
def run_loop(c, iterations):
    model = c["model"]
    for i in range(iterations):
        new = torch.zeros(1,128,requires_grad=True,device=c['device'])
        new = model(new)
        c['items'].insert(0,new.reshape(32,1))
        del c['items'][5:]
        yield torch.stack(c['items']).detach().cpu()
def roots(c):
    return c['items']
def terminal_loss(c):
    return torch.stack(c['items']).sum()
def cleanup(c):
    c['items'].clear()
