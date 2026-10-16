# Results layout

Generated evidence is grouped by phase. Each case is treated uniformly; summaries do not separate cases by origin.

- `phase_0_recon/`: environment record and authoritative static/measurement tool manifests.
- `phase_1_pilot/`: pilot measurements and static scans for C000-C005.
- `phase_2_runtime/`: paired runtime and CodeCarbon traces for C001-C009.
- `phase_3_baseline/`: four pinned static-tool scans for C001-C009.
- `phase_4_analysis/`: combined case summaries, tool results, adjudication, and gap analysis for C001-C009.

Run phase scripts from the repository root. Re-running a phase regenerates its output.

Descriptive plots can be regenerated from the recorded data with the plotting environment:

```bash
.venv-tools/bin/python scripts/phase_2_runtime/plot.py
.venv-tools/bin/python scripts/phase_4_analysis/plot.py
```

Phase 2 writes one four-panel graph/memory trajectory per case to `phase_2_runtime/plots/`. Phase 4 writes `memory_growth_summary.png` and `runtime_and_emissions_summary.png` to `phase_4_analysis/plots/`. The figures show measured values and repetition spread; they do not add detector outcomes or statistical conclusions.
