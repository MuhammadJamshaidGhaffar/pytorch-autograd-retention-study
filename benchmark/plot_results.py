"""Plot raw benchmark trajectories; shared protocols, separate mechanism observables."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
fig,axes=plt.subplots(2,5,figsize=(15,5.5),layout='constrained')
for col,cid in enumerate(['R001','R002','R003','R004','R005']):
    for variant,color in [('vulnerable','#b64136'),('fixed','#267b72')]:
        for run in range(1,4):
            rows=list(csv.DictReader((ROOT/'results/benchmark_runtime'/cid/f'{variant}_run{run:02}.csv').open()))
            x=[int(r['iteration']) for r in rows]
            for row,key in enumerate(['graph_nodes','allocated_bytes']):
                y=[int(r[key]) for r in rows]
                if row:y=[(v-y[39])/1024 for v in y]
                axes[row,col].plot(x,y,color=color,alpha=.6,label=variant if run==1 else None)
    axes[0,col].set_title(cid)
    for row in range(2):axes[row,col].grid(alpha=.2);axes[row,col].set_xlabel('Completed step')
axes[0,0].set_ylabel('Reachable autograd nodes');axes[1,0].set_ylabel('CUDA allocated Δ from step 40 (KiB)')
axes[0,0].legend(fontsize=8);fig.suptitle('Five public-source reductions • three runs per variant • graph growth is not always GPU growth')
for ext in ['png','pdf']:fig.savefig(ROOT/'results/figures'/f'benchmark_traces.{ext}',dpi=170)
