import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4,8,bias=False)
        self.lin_bis = nn.Linear(8,4,bias=False)
        self.register_buffer('tensor_bidon',torch.ones(self.lin.weight.shape))
    def forward(self,x):
        self.tensor_bidon.copy_(self.lin.weight)
        x = self.lin(x)
        x = self.lin_bis(x)
        return torch.matmul(x,self.tensor_bidon.t())
def build(device):
    model = MyModel().to(device)
    return {'model':model,'optimizer':torch.optim.Adam(model.parameters()),'device':device}
def run_loop(c, iterations):
    model = c["model"]
    for i in range(iterations):
        c['optimizer'].zero_grad(set_to_none=True)
        output = model(torch.randn(10,4,device=c['device']))
        target = torch.randn(10,8,device=c['device'])
        loss = nn.functional.mse_loss(output,target)
        loss.backward()
        gradients = torch.cat([p.grad.flatten() for p in model.parameters()]).detach().cpu()
        c['optimizer'].step()
        yield torch.cat([output.detach().cpu().flatten(),gradients])
def roots(c):
    return c['model'].tensor_bidon
def cleanup(c):
    c['model'].tensor_bidon = c['model'].tensor_bidon.detach()
