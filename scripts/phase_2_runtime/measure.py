"""Collect Phase 2 runtime and emissions evidence for cases 001 through 009.

Each vulnerable/fixed repetition runs in a fresh process and records graph,
memory, cleanup, and CodeCarbon measurements. case_000 is reserved for the
Phase 1 pilot. Each self-contained case exposes run_experiment(device, iterations,
seed, observer); the observer receives the live context after each step. This is
measurement code, not the proposed detector.
"""
import argparse,csv,gc,json,os,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.phase_1_pilot.measure import load,inspect_state

def worker(a):
    import torch
    from codecarbon import OfflineEmissionsTracker
    import importlib.metadata
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    case_dir=ROOT/'cases'/a.case
    mod=load(case_dir/(a.variant+'.py'),'subject')
    warm,_=mod.run_experiment(a.device,5,a.seed)
    mod.cleanup(warm)
    del warm
    gc.collect()
    torch.manual_seed(a.seed)
    if a.device=='cuda':
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    rows=[];start=time.perf_counter()
    out=Path(a.output)
    if not out.is_absolute(): out=ROOT/out
    out.parent.mkdir(parents=True,exist_ok=True)
    tracker=OfflineEmissionsTracker(country_iso_code='CAN',region='quebec',project_name=f'{a.case}-{a.variant}',
        tracking_mode='process',measure_power_secs=1,output_dir=str(out.parent),
        output_file=out.stem+'_codecarbon.csv',log_level='error')
    tracker.start()
    def observe(i,context):
        model=context['model']
        stats=inspect_state(mod.roots(context),model)
        if a.device=='cuda':torch.cuda.synchronize()
        rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        rows.append(dict(iteration=i+1,**stats,
            allocated_bytes=torch.cuda.memory_allocated() if a.device=='cuda' else 0,
            reserved_bytes=torch.cuda.memory_reserved() if a.device=='cuda' else 0,
            rss_bytes=rss))
    context,_=mod.run_experiment(a.device,a.iterations,a.seed,observer=observe,collect_outputs=False)
    model=context['model']
    emissions_kg=tracker.stop()
    assert len(rows)==a.iterations, 'Every case step must have exactly one sample'
    before=rows[-1]['allocated_bytes']
    mod.cleanup(context)
    gc.collect()
    roots_after=inspect_state(mod.roots(context),model)
    after=torch.cuda.memory_allocated() if a.device=='cuda' else 0
    with out.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    carbon_path=out.parent/(out.stem+'_codecarbon.csv')
    carbon=[]
    if carbon_path.exists():
        with carbon_path.open() as f: carbon=list(csv.DictReader(f))
    carbon_row=carbon[-1] if carbon else {}
    out.with_suffix('.json').write_text(json.dumps(dict(seed=a.seed,iterations=a.iterations,warmup=5,
        device=a.device,torch=torch.__version__,python=platform.python_version(),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,elapsed_seconds=time.perf_counter()-start,
        cleanup_graph_nodes=roots_after['graph_nodes'],released_by_cleanup_bytes=before-after,
        codecarbon_version=importlib.metadata.version('codecarbon'),carbon_tracking_mode='process',
        carbon_location_proxy='Montreal, Quebec (assumed calculation setting; not a statement of host location)',
        carbon_measure_power_secs=1,emissions_kg=emissions_kg,
        energy_consumed_kwh=float(carbon_row['energy_consumed']) if carbon_row.get('energy_consumed') else None,
        carbon_intensity_gco2e_per_kwh=float(carbon_row['emissions_rate']) if carbon_row.get('emissions_rate') else None,
        codecarbon_csv=str(carbon_path.relative_to(ROOT))),indent=2)+'\n')

def validate():
    import torch
    torch.set_num_threads(1)
    results=[]
    for folder in sorted(d for d in (ROOT/'cases').glob('case_*') if d.name != 'case_000'):
        trajectories=[]
        for variant in ('vulnerable','fixed'):
            torch.manual_seed(123)
            mod=load(folder/(variant+'.py'),'subject_'+variant)
            context,values=mod.run_experiment('cpu',12,123)
            terminal=mod.terminal_loss(context)
            if terminal is not None:
                terminal.backward()
            values.extend(p.detach().clone() for p in context['model'].parameters())
            values.extend(p.grad.detach().clone() for p in context['model'].parameters() if p.grad is not None)
            mod.cleanup(context)
            trajectories.append(values)
            del context,mod
            gc.collect()
        assert len(trajectories[0])==len(trajectories[1]), folder
        errors=[]
        for vulnerable,fixed in zip(*trajectories):
            torch.testing.assert_close(vulnerable,fixed,rtol=1e-5,atol=1e-6)
            errors.append(float((vulnerable-fixed).abs().max()))
        meta=json.loads((folder/'metadata.json').read_text())
        results.append(dict(case_id=meta['case_id'],status='PASS',max_abs_error=max(errors),steps=12,
            checks='per-step outputs; final parameters; available gradients and terminal-window gradient where applicable'))
    output=ROOT/'results/phase_4_analysis/numerical_validation.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='all');p.add_argument('--device',default='cuda',choices=['cuda','cpu']);p.add_argument('--iterations',type=int,default=120);p.add_argument('--seed',type=int,default=42);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--variant',choices=['vulnerable','fixed']);p.add_argument('--worker',action='store_true');p.add_argument('--validate',action='store_true');p.add_argument('--output');a=p.parse_args()
    if a.validate:validate();return
    if a.worker:worker(a);return
    cases=sorted(d for d in (ROOT/'cases').glob('case_*') if d.name != 'case_000') if a.case=='all' else [ROOT/'cases'/a.case]
    if any(folder.name == 'case_000' for folder in cases):
        p.error('case_000 is reserved for Phase 1 and cannot be measured in Phase 2.')
    for folder in cases:
        for run in range(1,a.repetitions+1):
            for variant in ('vulnerable','fixed'):
                out=ROOT/'results/phase_2_runtime'/folder.name/f'{variant}_run{run:02}.csv'
                print(folder.name,variant,run,flush=True)
                subprocess.run([sys.executable,__file__,'--worker','--case',folder.name,'--device',a.device,
                    '--iterations',str(a.iterations),'--seed',str(a.seed+run-1),'--variant',variant,
                    '--output',str(out)],check=True)
if __name__=='__main__':main()
