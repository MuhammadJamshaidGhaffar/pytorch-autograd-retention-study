"""C004: a forward hook stores graph-attached features across iterations."""
import torch
from torch import nn


def build(device):
    return {'model': nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128)).to(device), 'state': {}, 'device': device}


def _run_loop(context, batches):
    features = context['state'].setdefault('features', [])

    def capture(module, inputs, output):
        features.append(output)  # RETENTION_SINK -- BUG: hook retains each attached feature.

    handle = context['model'][1].register_forward_hook(capture)
    try:
        # --- CASE LOGIC START: vulnerable ---
        for batch in batches:
            output = context['model'](batch)
            # --- CASE LOGIC END ---
            snapshot = output.detach().cpu()
            del output, batch
            yield snapshot
    finally:
        handle.remove()


def roots(context):
    return context['state']


def terminal_loss(context):
    return None


def cleanup(context):
    context['state'].clear()


def make_batches(iterations, device, seed):
    """Create this case's deterministic inputs without an external workload file."""
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(iterations):
        yield torch.randn(32, 128, generator=generator, device=device)


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
