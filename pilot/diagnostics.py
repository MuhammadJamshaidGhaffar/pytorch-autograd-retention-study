"""Complementary profiler/snapshot experiment; never scored as static recall."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import subprocess
import sys

from measure import ROOT, load
from portable import data as portable_data, project_frames


def worker(case, variant, iterations):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(42)
    d=ROOT/'cases'/case
    mod=load(d/(variant+'.py'),'diagnostic_case')
    workload=load(d/'workload.py','diagnostic_workload')
    model=mod.make_model('cuda')
    warm={}
    mod.run_loop(model,workload.get_workload(5,'cuda'),warm,lambda _:None)
    warm.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    out=ROOT/'results/pilot_diagnostics'/case;out.mkdir(parents=True,exist_ok=True)
    # Collect snapshots separately: profiler record_shapes can itself retain tensor refs.
    torch.cuda.memory._record_memory_history(enabled='all',context='all',stacks='all',max_entries=50000)
    state={};samples=[]
    def sample(_):
        torch.cuda.synchronize()
        if not samples:
            with gzip.open(out/f'{variant}_early_snapshot.pickle.gz','wb') as f:
                pickle.dump(portable_data(torch.cuda.memory._snapshot()),f)
        samples.append(torch.cuda.memory_allocated())
    mod.run_loop(model,workload.get_workload(iterations,'cuda'),state,sample)
    snapshot=torch.cuda.memory._snapshot()
    with gzip.open(out/f'{variant}_final_snapshot.pickle.gz','wb') as f:
        pickle.dump(portable_data(snapshot),f)
    blocks=[b for seg in snapshot['segments'] for b in seg['blocks'] if b['state']=='active_allocated']
    relevant=[b for b in blocks if any(f.get('filename','').endswith(variant+'.py') for f in b.get('frames',[]))]
    summary={'case':case,'variant':variant,'iterations':iterations,'allocated_first':samples[0],
             'allocated_last':samples[-1],'active_blocks':len(blocks),'active_bytes':sum(b['size'] for b in blocks),
             'blocks_with_case_source_frame':len(relevant),
             'example_source_frames':project_frames(relevant[0]['frames']) if relevant else [],
             'source_frame_note':'Only project Python frames shown; unresolved native frames omitted.',
             'measurement_note':'Snapshot bytes cover all active allocations. Profiler memory is inclusive and overlaps across operators; do not sum. Separate passes.',
             'retention_relation_automatically_explained':False,
             'interpretation':'Allocation stacks identify producers; linking them to persistent owners requires manual inspection.'}
    torch.cuda.memory._record_memory_history(enabled=None)
    state.clear()
    torch.cuda.empty_cache()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                profile_memory=True,with_stack=True,record_shapes=True) as prof:
        mod.run_loop(model,workload.get_workload(iterations,'cuda'),state,lambda _:prof.step())
    trace=out/f'{variant}_profiler.json'
    prof.export_chrome_trace(str(trace))
    trace.write_text(json.dumps(portable_data(json.loads(trace.read_text()))))
    with trace.open('rb') as inp,gzip.open(str(trace)+'.gz','wb') as dest:
        import shutil
        shutil.copyfileobj(inp,dest)
    trace.unlink()
    events=[]
    for e in prof.key_averages():
        if e.cpu_memory_usage or e.device_memory_usage:
            events.append({'operator':e.key,'calls':e.count,'cpu_memory_usage':e.cpu_memory_usage,
                           'device_memory_usage':e.device_memory_usage})
    summary['profiler_allocating_operators']=events
    summary['profiler_events_with_stack']=sum(bool(e.stack) for e in prof.events())
    (out/f'{variant}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    p.add_argument('--case');p.add_argument('--variant');p.add_argument('--iterations',type=int,default=20)
    a=p.parse_args()
    if a.worker:worker(a.case,a.variant,a.iterations);return
    for case in sorted((ROOT/'cases').glob('case_*')):
        for variant in ['vulnerable','fixed']:
            print('diagnostics',case.name,variant,flush=True)
            subprocess.run([sys.executable,__file__,'--worker','--case',case.name,'--variant',variant,
                            '--iterations',str(a.iterations)],check=True)

if __name__=='__main__':main()
