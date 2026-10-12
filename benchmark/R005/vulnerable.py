import torch
from torch import nn

def build(device):
    return {'model':nn.Linear(128,32).to(device),'window':torch.zeros(5,32,1,requires_grad=True,device=device),'device':device}
def run_loop(c, iterations):
    model = c["model"]
    for i in range(iterations):
        new = torch.zeros(1,128,requires_grad=True,device=c['device'])
        new = model(new)
        c['window'] = torch.cat((new.unsqueeze(-1),c['window'][:-1]),dim=0)
        yield c['window'].detach().cpu()
def roots(c):
    return c['window']
def terminal_loss(c):
    return c['window'].sum()
def cleanup(c):
    c['window'] = c['window'].detach()
