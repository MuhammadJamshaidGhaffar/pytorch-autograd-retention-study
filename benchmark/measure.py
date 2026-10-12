"""Phase 2 evidence collection; not a detector. Separate processes for each run."""
import argparse,csv,gc,json,os,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pilot.measure import load,inspect_state

def worker(a):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    mod=load(ROOT/'benchmark'/a.case/(a.variant+'.py'),'subject')
    warm=mod.build(a.device)
    for value in mod.run_loop(warm,5): pass
    del value
    mod.cleanup(warm);del warm;gc.collect()
    torch.manual_seed(a.seed)
    c=mod.build(a.device)
    if a.device=='cuda':
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    rows=[];start=time.perf_counter()
    for i, value in enumerate(mod.run_loop(c,a.iterations)):
        del value
        stats=inspect_state(mod.roots(c),c['model'])
        if a.device=='cuda':torch.cuda.synchronize()
        rows.append(dict(iteration=i+1,**stats,allocated_bytes=torch.cuda.memory_allocated() if a.device=='cuda' else 0,reserved_bytes=torch.cuda.memory_reserved() if a.device=='cuda' else 0,rss_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')))
    before=rows[-1]['allocated_bytes'];mod.cleanup(c);gc.collect()
    after_graph=inspect_state(mod.roots(c),c['model'])
    after=torch.cuda.memory_allocated() if a.device=='cuda' else 0
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    out.with_suffix('.json').write_text(json.dumps(dict(seed=a.seed,iterations=a.iterations,warmup=5,device=a.device,torch=torch.__version__,python=platform.python_version(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,elapsed_seconds=time.perf_counter()-start,cleanup_graph_nodes=after_graph['graph_nodes'],released_by_cleanup_bytes=before-after),indent=2)+'\n')

def validate():
    import torch
    torch.set_num_threads(1)
    results=[]
    for folder in sorted((ROOT/'benchmark').glob('R*')):
        trajectories=[]
        for variant in ['vulnerable','fixed']:
            torch.manual_seed(123)
            mod=load(folder/(variant+'.py'),'subject_'+variant)
            c=mod.build('cpu');values=list(mod.run_loop(c,12))
            if hasattr(mod,'terminal_loss'):
                mod.terminal_loss(c).backward()
                values.append(torch.cat([p.grad.flatten() for p in c['model'].parameters()]))
            values.extend(p.detach().clone() for p in c['model'].parameters())
            trajectories.append(values)
            mod.cleanup(c);del c;gc.collect()
        errors=[]
        for v,f in zip(*trajectories):
            torch.testing.assert_close(v,f,rtol=1e-5,atol=1e-6)
            errors.append(float((v-f).abs().max()))
        results.append(dict(case_id=folder.name,status='PASS',max_abs_error=max(errors),steps=12,checks='per-step outputs and gradients where training; final parameters; R005 terminal window gradient'))
    (ROOT/'results/benchmark_numerical_validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='all');p.add_argument('--device',default='cuda',choices=['cuda','cpu']);p.add_argument('--iterations',type=int,default=120);p.add_argument('--seed',type=int,default=42);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--variant',choices=['vulnerable','fixed']);p.add_argument('--worker',action='store_true');p.add_argument('--validate',action='store_true');p.add_argument('--output');a=p.parse_args()
    if a.validate:validate();return
    if a.worker:worker(a);return
    cases=sorted((ROOT/'benchmark').glob('R*')) if a.case=='all' else [ROOT/'benchmark'/a.case]
    for folder in cases:
        for run in range(1,a.repetitions+1):
            for variant in ['vulnerable','fixed']:
                out=ROOT/'results/benchmark_runtime'/folder.name/f'{variant}_run{run:02}.csv'
                print(folder.name,variant,run,flush=True)
                subprocess.run([sys.executable,__file__,'--worker','--case',folder.name,'--device',a.device,'--iterations',str(a.iterations),'--seed',str(a.seed+run-1),'--variant',variant,'--output',str(out)],check=True)
if __name__=='__main__':main()
