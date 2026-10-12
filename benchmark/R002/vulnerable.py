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
        self._total += value

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
    return {'model':model, 'meter':MeterBuffer(), 'optimizer':torch.optim.SGD(model.parameters(), lr=0.001), 'device':device}
def run_loop(c, iterations):
    model = c["model"]
    for i in range(iterations):
        c['optimizer'].zero_grad(set_to_none=True)
        output = model(torch.randn(16,128,device=c['device']))
        loss = output.square().mean()
        loss.backward()
        gradient = torch.cat([p.grad.flatten() for p in model.parameters()]).detach().cpu()
        c['optimizer'].step()
        c['meter'].update({'loss':loss})
        yield torch.cat([loss.detach().cpu().reshape(1),gradient])
def roots(c):
    return c['meter']
def cleanup(c):
    c['meter'].reset()
