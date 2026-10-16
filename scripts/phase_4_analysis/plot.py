"""Plot descriptive Phase 4 summaries of recorded runtime measurements.

The script reads Phase 4 runtime_ground_truth.csv and the Phase 2 per-run JSON
metadata. It creates paired vulnerable/fixed visual summaries for memory,
graph, elapsed time, and CodeCarbon estimates. Dots show individual runs and
large markers show medians. These plots are descriptive representations only;
they do not calculate detector metrics or make statistical conclusions.

Run after Phase 2 and Phase 4 summarization:
    .venv-tools/bin/python scripts/phase_4_analysis/plot.py
"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PHASE2 = ROOT / 'results' / 'phase_2_runtime'
PHASE4 = ROOT / 'results' / 'phase_4_analysis'
OUTPUT = PHASE4 / 'plots'
VARIANTS = ('vulnerable', 'fixed')
COLORS = {'vulnerable': '#c44e52', 'fixed': '#4c72b0'}


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def median_range(axis, cases, run_values, ylabel, title, scale=1.0, unit_note=None):
    positions = list(range(len(cases)))
    offsets = {'vulnerable': -0.16, 'fixed': 0.16}
    for variant in VARIANTS:
        color = COLORS[variant]
        for index, case_id in enumerate(cases):
            values = [value / scale for value in run_values[(case_id, variant)]]
            x = positions[index] + offsets[variant]
            # Small deterministic separation keeps repeated points visible.
            jitter = (-0.045, 0.0, 0.045)
            axis.scatter([x + jitter[i] for i in range(len(values))], values,
                         color=color, alpha=0.72, s=22, zorder=3)
            median = statistics.median(values)
            axis.scatter([x], [median], color=color, edgecolor='white', linewidth=0.7,
                         marker='D', s=58, zorder=4,
                         label=variant.capitalize() if index == 0 else None)
    axis.set_xticks(positions, cases)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    axis.grid(axis='y', alpha=0.25)
    if unit_note:
        axis.text(0.01, 0.98, unit_note, transform=axis.transAxes,
                  va='top', ha='left', fontsize=8, color='#555555')


def main():
    truth_path = PHASE4 / 'runtime_ground_truth.csv'
    if not truth_path.exists():
        raise FileNotFoundError(f'Run Phase 4 summarization first: {truth_path.relative_to(ROOT)}')
    rows = read_csv(truth_path)
    cases = sorted({row['case_id'] for row in rows})
    if not cases:
        raise ValueError(f'No runtime rows found in {truth_path.relative_to(ROOT)}')
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Per-run memory and graph quantities are read directly from Phase 4's
    # paired summary, which is derived from the original Phase 2 traces.
    fields = {
        'allocated_growth_bytes': 'Allocated GPU memory change, iterations 40–120 (MiB)',
        'saved_activation_bytes_at_120': 'Saved activations at iteration 120 (MiB)',
        'graph_nodes_at_120': 'Autograd graph nodes at iteration 120',
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for axis, (field, ylabel) in zip(axes, fields.items()):
        values = {(case, variant): [float(row[field]) for row in rows
                                    if row['case_id'] == case and row['variant'] == variant]
                  for case in cases for variant in VARIANTS}
        memory_metric = field in {'allocated_growth_bytes', 'saved_activation_bytes_at_120'}
        median_range(axis, cases, values, ylabel, ylabel,
                     scale=1024 ** 2 if memory_metric else 1)
    axes[-1].set_xlabel('Case')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=2, frameon=False)
    fig.suptitle('Phase 4 — recorded memory and graph measurements', y=0.995, fontsize=14)
    fig.text(0.5, 0.008,
             'Each point is one of 3 runs; diamonds mark medians. '
             'Memory growth is the measured change from iteration 40 to 120.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    memory_path = OUTPUT / 'memory_growth_summary.png'
    fig.savefig(memory_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Elapsed time and emissions are recorded in each run's JSON sidecar.
    runtime = {(case, variant): [] for case in cases for variant in VARIANTS}
    emissions = {(case, variant): [] for case in cases for variant in VARIANTS}
    for case in cases:
        case_folder = PHASE2 / f'case_{int(case[1:]):03}'
        for variant in VARIANTS:
            for run_id in range(1, 4):
                path = case_folder / f'{variant}_run{run_id:02}.json'
                if not path.exists():
                    raise FileNotFoundError(f'Missing run metadata: {path.relative_to(ROOT)}')
                metadata = json.loads(path.read_text())
                runtime[(case, variant)].append(float(metadata['elapsed_seconds']))
                emissions[(case, variant)].append(float(metadata['emissions_kg']))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    median_range(axes[0], cases, runtime, 'Elapsed time (seconds)',
                 'Recorded elapsed time per run')
    median_range(axes[1], cases, emissions, 'Estimated emissions (µg CO₂e)',
                 'CodeCarbon per-process estimate', scale=1e-9,
                 unit_note='Converted from recorded kg CO₂e for readability')
    axes[1].set_xlabel('Case')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=2, frameon=False)
    fig.suptitle('Phase 4 — recorded runtime and CodeCarbon estimates',
                 y=0.995, fontsize=14)
    fig.text(0.5, 0.008,
             'Each point is one of 3 runs; diamonds mark medians. '
             'Emissions are model estimates from the recorded CodeCarbon output.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    runtime_path = OUTPUT / 'runtime_and_emissions_summary.png'
    fig.savefig(runtime_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    print(f'Generated descriptive Phase 4 plots in {OUTPUT.relative_to(ROOT)}:')
    print(f'  {memory_path.name}')
    print(f'  {runtime_path.name}')


if __name__ == '__main__':
    main()
