import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4,8,bias=False)
        self.lin_bis = nn.Linear(8,4,bias=False)
        self.register_buffer('tensor_bidon',torch.ones(self.lin.weight.shape))
    def forward(self,x):
        self.tensor_bidon = self.tensor_bidon.detach()  # FIX: clear prior history before the new copy.
        self.tensor_bidon.copy_(self.lin.weight)
        x = self.lin(x)
        x = self.lin_bis(x)
        return torch.matmul(x,self.tensor_bidon.t())
def build(device):
    model = MyModel().to(device)
    return {'model': model, 'optimizer': torch.optim.Adam(model.parameters()), 'device': device}


def _run_loop(context, batches):
    model = context['model']
    # --- CASE LOGIC START: fixed ---
    for inputs, target in batches:
        context['optimizer'].zero_grad(set_to_none=True)
        output = model(inputs)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        gradients = torch.cat([parameter.grad.flatten() for parameter in model.parameters()]).detach().cpu()
        context['optimizer'].step()
        # --- CASE LOGIC END ---
        snapshot = torch.cat([output.detach().cpu().flatten(), gradients])
        del output, target, loss, gradients, inputs
        yield snapshot


def roots(context):
    return context['model'].tensor_bidon


def terminal_loss(context):
    return None


def cleanup(context):
    context['model'].tensor_bidon = context['model'].tensor_bidon.detach()


def make_batches(iterations, device, seed):
    """Create this case's deterministic inputs without an external workload file."""
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(iterations):
        inputs = torch.randn(10, 4, generator=generator, device=device)
        target = torch.randn(10, 8, generator=generator, device=device)
        yield inputs, target


def run_experiment(device="cpu", iterations=20, seed=42, observer=None, collect_outputs=True):
    """Run directly or let the measurement runner observe each step.

    Returns the live case context and detached CPU output snapshots. The optional
    observer receives (zero_based_step, context) after each case step.
    """
    torch.manual_seed(seed)
    context = build(device)
    snapshots = []
    for step, snapshot in enumerate(_run_loop(context, make_batches(iterations, device, seed))):
        if collect_outputs:
            snapshots.append(snapshot)
        if observer is not None:
            observer(step, context)
    return context, snapshots


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__ or "Run this autograd retention case")
    parser.add_argument("--device", default="cpu", help="torch device, for example cpu or cuda")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    context, snapshots = run_experiment(args.device, args.iterations, args.seed)
    print(json.dumps({"steps": len(snapshots), "device": args.device, "seed": args.seed}))
    cleanup(context)


if __name__ == "__main__":
    main()
