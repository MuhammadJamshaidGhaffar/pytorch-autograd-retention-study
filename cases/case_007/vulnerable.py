# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Extracted from YOLOX (Apache-2.0); see provenance.json and LICENSE.upstream.
import functools
from collections import defaultdict, deque
import numpy as np
import torch
from torch import nn
class AverageMeter:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=50):
        self._deque = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value  # RETENTION_SINK -- BUG: running total retains the loss graph.

    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)

    @property
    def avg(self):
        # if deque is empty, nan will be returned.
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def global_avg(self):
        return self._total / max(self._count, 1e-5)

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None

    @property
    def total(self):
        return self._total

    def reset(self):
        self._deque.clear()
        self._total = 0.0
        self._count = 0

    def clear(self):
        self._deque.clear()

class MeterBuffer(defaultdict):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter, window_size=window_size)
        super().__init__(factory)

    def reset(self):
        for v in self.values():
            v.reset()

    def get_filtered_meter(self, filter_key="time"):
        return {k: v for k, v in self.items() if filter_key in k}

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()

def build(device):
    model = nn.Linear(128, 32).to(device)
    return {'model': model, 'meter': MeterBuffer(), 'optimizer': torch.optim.SGD(model.parameters(), lr=0.001), 'device': device}


def _run_loop(context, batches):
    model = context['model']
    # --- CASE LOGIC START: vulnerable ---
    for batch in batches:
        context['optimizer'].zero_grad(set_to_none=True)
        output = model(batch)
        loss = output.square().mean()
        loss.backward()
        gradient = torch.cat([parameter.grad.flatten() for parameter in model.parameters()]).detach().cpu()
        context['optimizer'].step()
        context['meter'].update({'loss': loss})
        # --- CASE LOGIC END ---
        snapshot = torch.cat([loss.detach().cpu().reshape(1), gradient])
        del output, loss, gradient, batch
        yield snapshot


def roots(context):
    return context['meter']


def terminal_loss(context):
    return None


def cleanup(context):
    context['meter'].reset()


def make_batches(iterations, device, seed):
    """Create this case's deterministic inputs without an external workload file."""
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(iterations):
        yield torch.randn(16, 128, generator=generator, device=device)


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
