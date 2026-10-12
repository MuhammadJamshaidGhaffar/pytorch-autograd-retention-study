"""Runtime ground truth, not a static detector. One fresh process per pair member/run."""
import argparse
import csv
from collections import deque
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensors_in(value, seen=None):
    import torch
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item, seen)
    elif isinstance(value, (list, tuple, deque)):
        for item in value:
            yield from tensors_in(item, seen)
    elif hasattr(value, '__dict__'):
        yield from tensors_in(vars(value), seen)


def storage(t):
    s = t.untyped_storage()
    return (str(t.device), s.data_ptr()), s.nbytes()


def inspect_state(state, model):
    """Count reachable nodes and unique saved storages; release inspection refs on return.

    _saved_* introspection is version-specific and corroborates CUDA allocation,
    not a portable total graph-memory estimator. Parameters and output payload
    are excluded from saved_activation_bytes. Python graph-node memory is separate.
    """
    import torch
    roots = list(tensors_in(state))
    payload = dict(storage(t) for t in roots)
    excluded = set(payload) | {storage(p)[0] for p in model.parameters()}
    pending = [t.grad_fn for t in roots if t.grad_fn is not None]
    visited = set()
    saved = {}
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(n for n, _ in node.next_functions if n is not None)
        for attr in dir(node):
            if attr.startswith('_saved_'):
                try:
                    value = getattr(node, attr)
                except RuntimeError:
                    continue  # Backward may have freed saved tensors.
                for tensor in tensors_in(value):
                    key, size = storage(tensor)
                    if key not in excluded:
                        saved[key] = size
    return {'retained_tensors': len(roots),
            'attached_roots': sum(t.grad_fn is not None for t in roots),
            'graph_nodes': len(visited), 'payload_bytes': sum(payload.values()),
            'saved_activation_bytes': sum(saved.values())}


def worker(args):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; this is NOT a clean runtime result')
    folder = ROOT / args.case
    mod = load(folder / (args.variant + '.py'), 'case_module')
    workload = load(folder / 'workload.py', 'workload_module')
    model = mod.make_model(args.device)
    # Warm up with discarded state, then measure exactly once after each completed step.
    warm = {}
    mod.run_loop(model, workload.get_workload(args.warmup, args.device, args.seed), warm, lambda _: None)
    warm.clear()
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    state, records = {}, []
    start = time.perf_counter()
    def observe(current):
        stats = inspect_state(current, model)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        # Current RSS, not ru_maxrss (which is a historical peak).
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        records.append({'iteration':len(records)+1, **stats,
                        'allocated_bytes':torch.cuda.memory_allocated() if args.device == 'cuda' else '',
                        'reserved_bytes':torch.cuda.memory_reserved() if args.device == 'cuda' else '',
                        'rss_bytes':rss})
    mod.run_loop(model, workload.get_workload(args.iterations, args.device, args.seed), state, observe)
    assert len(records) == args.iterations, 'Every workload step must have exactly one sample'
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=records[0], lineterminator="\n");writer.writeheader();writer.writerows(records)
    before = torch.cuda.memory_allocated() if args.device == 'cuda' else None
    state.clear()
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.synchronize()
    after = torch.cuda.memory_allocated() if args.device == 'cuda' else None
    env = {'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
           'device':args.device,'gpu':torch.cuda.get_device_name() if args.device == 'cuda' else None,
           'os':platform.platform(),'seed':args.seed,'iterations':args.iterations,'warmup':args.warmup,
           'elapsed_seconds_including_observation':time.perf_counter()-start,
           'allocated_before_cleanup':before,'allocated_after_cleanup':after,
           'released_by_cleanup_bytes':before-after if before is not None else None}
    out.with_suffix('.json').write_text(json.dumps(env,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',default='all',help='case_002 or all')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--iterations',type=int,default=80)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--variant',choices=['vulnerable','fixed'])
    p.add_argument('--output')
    a=p.parse_args()
    if a.iterations < 2 or a.warmup < 0 or a.repetitions < 1:
        p.error('Need iterations >= 2, warmup >= 0, repetitions >= 1')
    if a.worker:
        worker(a);return
    cases=sorted(ROOT.glob('cases/case_*')) if a.case=='all' else [ROOT / (a.case if '/' in a.case else 'cases/'+a.case)]
    for case in cases:
        for run in range(1,a.repetitions+1):
            for variant in ['vulnerable','fixed']:
                output=ROOT/'results/pilot_runtime'/case.name/f'{variant}_run{run:02}.csv'
                cmd=[sys.executable,__file__,'--worker','--case',str(case.relative_to(ROOT)),
                     '--variant',variant,'--device',a.device,'--seed',str(a.seed+run-1),
                     '--iterations',str(a.iterations),'--warmup',str(a.warmup),'--output',str(output)]
                print(case.name,variant,run,flush=True)
                subprocess.run(cmd,check=True)

if __name__=='__main__':
    main()
