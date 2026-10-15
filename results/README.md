# Results layout

Generated evidence is grouped by phase. Each case is treated uniformly; summaries do not separate cases by origin.

- `phase_0_recon/`: environment record and authoritative static/measurement tool manifests.
- `phase_1_pilot/`: pilot measurements and static scans for C000-C005.
- `phase_2_runtime/`: paired runtime and CodeCarbon traces for C001-C009.
- `phase_3_baseline/`: four pinned static-tool scans for C001-C009.
- `phase_4_analysis/`: combined case summaries, tool results, adjudication, and gap analysis for C001-C009.

Run phase scripts from the repository root. Re-running a phase regenerates its output.
