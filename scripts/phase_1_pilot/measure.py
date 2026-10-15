"""Run the Phase 1 pilot runtime on cases C000-C005.

For every vulnerable/fixed repetition this script samples reachable autograd
nodes, saved activation storage, CUDA allocation, process RSS, and cleanup
behavior in a fresh process. It records experimental evidence; it is not a
static detector. Phase 2 cases use scripts/phase_2_runtime/measure.py for cases C001-C009.
"""
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

ROOT = Path(__file__).resolve().parents[2]


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
    from codecarbon import OfflineEmissionsTracker
    import importlib.metadata
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; this is NOT a clean runtime result')
    folder = ROOT / args.case
    mod = load(folder / (args.variant + '.py'), 'case_module')
    # Warm up with a fresh context, then release it before measurement.
    warm, _ = mod.run_experiment(args.device, args.warmup, args.seed)
    mod.cleanup(warm)
    del warm
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    records = []
    start = time.perf_counter()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tracker = OfflineEmissionsTracker(country_iso_code='CAN', region='quebec',
        project_name=f'{args.case}-{args.variant}', tracking_mode='process',
        measure_power_secs=1, output_dir=str(out.parent),
        output_file=out.stem+'_codecarbon.csv', log_level='error')
    tracker.start()
    def observe(step, context):
        model = context['model']
        stats = inspect_state(mod.roots(context), model)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        # Current RSS, not ru_maxrss (which is a historical peak).
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        records.append({'iteration':step+1, **stats,
                        'allocated_bytes':torch.cuda.memory_allocated() if args.device == 'cuda' else '',
                        'reserved_bytes':torch.cuda.memory_reserved() if args.device == 'cuda' else '',
                        'rss_bytes':rss})
    context, _ = mod.run_experiment(args.device, args.iterations, args.seed,
                                    observer=observe, collect_outputs=False)
    model = context['model']
    emissions_kg = tracker.stop()
    assert len(records) == args.iterations, 'Every workload step must have exactly one sample'
    with out.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=records[0], lineterminator="\n");writer.writeheader();writer.writerows(records)
    carbon_path = out.parent/(out.stem+'_codecarbon.csv')
    carbon_rows = list(csv.DictReader(carbon_path.open())) if carbon_path.exists() else []
    carbon_row = carbon_rows[-1] if carbon_rows else {}
    before = torch.cuda.memory_allocated() if args.device == 'cuda' else None
    mod.cleanup(context)
    cleanup_graph_nodes = inspect_state(mod.roots(context), model)['graph_nodes']
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.synchronize()
    after = torch.cuda.memory_allocated() if args.device == 'cuda' else None
    env = {'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
           'device':args.device,'gpu':torch.cuda.get_device_name() if args.device == 'cuda' else None,
           'os':platform.platform(),'seed':args.seed,'iterations':args.iterations,'warmup':args.warmup,
           'elapsed_seconds_including_observation':time.perf_counter()-start,
           'allocated_before_cleanup':before,'allocated_after_cleanup':after,
           'released_by_cleanup_bytes':before-after if before is not None else None,
           'cleanup_graph_nodes':cleanup_graph_nodes,
           'codecarbon_version':importlib.metadata.version('codecarbon'),
           'carbon_tracking_mode':'process',
           'carbon_location_proxy':'Montreal, Quebec (assumed calculation setting; not a statement of host location)',
           'emissions_kg':emissions_kg,
           'energy_consumed_kwh':float(carbon_row['energy_consumed']) if carbon_row.get('energy_consumed') else None,
           'codecarbon_csv':str(carbon_path.relative_to(ROOT))}
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
    cases=sorted((ROOT/'cases').glob('case_00[0-5]')) if a.case=='all' else [ROOT / (a.case if '/' in a.case else 'cases/'+a.case)]
    if any(case.name not in {f'case_{i:03}' for i in range(6)} for case in cases):
        p.error('Phase 1 covers cases C000-C005; use scripts/phase_2_runtime/measure.py for C001-C009 study runs.')
    for case in cases:
        for run in range(1,a.repetitions+1):
            for variant in ['vulnerable','fixed']:
                output=ROOT/'results/phase_1_pilot/runtime'/case.name/f'{variant}_run{run:02}.csv'
                cmd=[sys.executable,__file__,'--worker','--case',str(case.relative_to(ROOT)),
                     '--variant',variant,'--device',a.device,'--seed',str(a.seed+run-1),
                     '--iterations',str(a.iterations),'--warmup',str(a.warmup),'--output',str(output)]
                print(case.name,variant,run,flush=True)
                subprocess.run(cmd,check=True)

if __name__=='__main__':
    main()
