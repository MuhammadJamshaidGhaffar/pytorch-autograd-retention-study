import torch
from torch import nn

def build(device):
    return {'model': nn.Linear(128,32).to(device), 'window': torch.zeros(5,32,1,requires_grad=True,device=device), 'device': device}


def _run_loop(context, batches):
    # --- CASE LOGIC START: vulnerable ---
    for new in batches:
        new = context['model'](new)
        context['window'] = torch.cat((new.unsqueeze(-1), context['window'][:-1]), dim=0)  # RETENTION_SINK -- BUG: links this window to the prior graph.
        # --- CASE LOGIC END ---
        snapshot = context['window'].detach().cpu()
        del new
        yield snapshot


def roots(context):
    return context['window']


def terminal_loss(context):
    return context['window'].sum()


def cleanup(context):
    context['window'] = context['window'].detach()


def make_batches(iterations, device, seed):
    """Create this case's deterministic inputs without an external workload file."""
    for _ in range(iterations):
        yield torch.zeros(1, 128, requires_grad=True, device=device)


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
