"""Export pilot figures from preserved CSVs, without rerunning workloads."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
fig,axes=plt.subplots(2,4,figsize=(14,6.6),sharex=True)
for i,ax in enumerate(axes.flat):
    if i==7:ax.axis('off');continue
    for variant,color in [('vulnerable','#b33b37'),('fixed','#2579a9')]:
        for run in range(1,4):
            path=ROOT/f'results/pilot_runtime/case_{i:03}/{variant}_run{run:02}.csv'
            rows=list(csv.DictReader(path.open()))
            initial=int(rows[0]['allocated_bytes'])
            ax.plot([int(r['iteration']) for r in rows],[(int(r['allocated_bytes'])-initial)/2**20 for r in rows],
                    color=color,alpha=0.7,label=variant if run==1 else None)
    ax.set_title(f'C{i:03}');ax.set_xlabel('Completed iteration');ax.set_ylabel('Allocated growth (MiB)')
    ax.grid(alpha=0.2)
axes[0,0].legend(frameon=False)
fig.suptitle('GPU allocation after each step: three repetitions per variant\nDetached controls may still accumulate output payload',fontsize=13)
fig.tight_layout()
out=ROOT/'results/figures';out.mkdir(exist_ok=True)
fig.savefig(out/'pilot_memory_traces.png',dpi=180)
fig.savefig(out/'pilot_memory_traces.pdf')
