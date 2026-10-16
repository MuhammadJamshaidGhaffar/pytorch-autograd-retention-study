"""Plot descriptive Phase 2 runtime traces for cases C001-C009.

This script reads existing per-iteration CSV traces and produces one figure per
case. Lines show the median of three repetitions and shaded bands show the
observed minimum-to-maximum range. It does not rerun experiments or classify
cases; it only makes the recorded graph, activation, memory, and RSS measures
easier to inspect.

Run from the repository root with the plotting dependency already pinned in
baselines/requirements.lock:
    .venv-tools/bin/python scripts/phase_2_runtime/plot.py
"""
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'phase_2_runtime'
OUTPUT = RESULTS / 'plots'
VARIANTS = ('vulnerable', 'fixed')
COLORS = {'vulnerable': '#c44e52', 'fixed': '#4c72b0'}
METRICS = (
    ('graph_nodes', 'Autograd graph nodes', 1.0),
    ('saved_activation_bytes', 'Saved activations (MiB)', 1024 ** 2),
    ('allocated_bytes', 'GPU allocated change from step 1 (MiB)', 1024 ** 2),
    ('rss_bytes', 'Process RSS change from step 1 (MiB)', 1024 ** 2),
)


def read_trace(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def load_runs(case_dir, variant):
    runs = []
    for run_id in range(1, 4):
        path = case_dir / f'{variant}_run{run_id:02}.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing Phase 2 trace: {path.relative_to(ROOT)}')
        rows = read_trace(path)
        if not rows:
            raise ValueError(f'Empty Phase 2 trace: {path.relative_to(ROOT)}')
        runs.append(rows)
    lengths = {len(rows) for rows in runs}
    if len(lengths) != 1:
        raise ValueError(f'Unequal repetition lengths for {case_dir.name} {variant}')
    return runs


def metric_values(rows, key, scale):
    values = [float(row[key]) / scale for row in rows]
    if key in {'allocated_bytes', 'rss_bytes'}:
        baseline = values[0]
        values = [value - baseline for value in values]
    return values


def main():
    cases = sorted(path for path in RESULTS.glob('case_*') if path.is_dir())
    if not cases:
        raise FileNotFoundError(f'No Phase 2 case traces found under {RESULTS}')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated = []

    for case_dir in cases:
        data = {variant: load_runs(case_dir, variant) for variant in VARIANTS}
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.ravel()
        for axis, (key, label, scale) in zip(axes, METRICS):
            for variant in VARIANTS:
                runs = [metric_values(rows, key, scale) for rows in data[variant]]
                iterations = list(range(1, len(runs[0]) + 1))
                medians = [statistics.median(row[i] for row in runs)
                           for i in range(len(iterations))]
                minimums = [min(row[i] for row in runs) for i in range(len(iterations))]
                maximums = [max(row[i] for row in runs) for i in range(len(iterations))]
                color = COLORS[variant]
                axis.plot(iterations, medians, color=color, label=variant.capitalize(),
                          linewidth=1.8)
                axis.fill_between(iterations, minimums, maximums, color=color, alpha=0.16,
                                  linewidth=0)
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
        axes[2].set_xlabel('Measured iteration')
        axes[3].set_xlabel('Measured iteration')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96),
                   ncol=2, frameon=False)
        fig.suptitle(f'{case_dir.name.replace("case_", "C")} — recorded runtime traces',
                     y=0.99, fontsize=14)
        fig.text(0.5, 0.01,
                 'Median across 3 repetitions; shaded band is observed min–max. '
                 'GPU allocation and RSS are changes from measured iteration 1.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.045, 1, 0.92))
        output = OUTPUT / f'{case_dir.name}_runtime_memory.png'
        fig.savefig(output, dpi=180, bbox_inches='tight')
        plt.close(fig)
        generated.append(output)

    print(f'Generated {len(generated)} descriptive Phase 2 plots in '
          f'{OUTPUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
