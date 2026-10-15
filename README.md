# PyTorch autograd retention study

This study tests how well static-analysis tools detect PyTorch code that unintentionally retains autograd graphs. Each case pairs a vulnerable program with a corrected version.

## Quickstart

Run from the repository root. You need CUDA-enabled PyTorch, CodeQL on `PATH`, and the pinned dependencies. The runtime environment must be able to import the CUDA-enabled PyTorch installation.

Create the two local environments:

```bash
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install -r baselines/requirements.lock

python3 -m venv --system-site-packages .venv-runtime
.venv-runtime/bin/python -m pip install -r baselines/codecarbon_requirements.lock
```

Run each phase in order. Phase 0 records the environment and checks the baseline versions:

```bash
.venv-tools/bin/python scripts/phase_0_recon/verify_tools.py
.venv-runtime/bin/python scripts/phase_0_recon/record_environment.py
```

Phase 1 runs the pilot measurements, static scans, and summary:

```bash
python3 scripts/phase_1_pilot/run_static.py --bin-dir .venv-tools/bin
.venv-runtime/bin/python scripts/phase_1_pilot/measure.py --case all
.venv-runtime/bin/python scripts/phase_1_pilot/summarize.py
```

Check numerical equivalence, then collect the Phase 2 runtime traces:

```bash
.venv-runtime/bin/python scripts/phase_4_analysis/validate_numerics.py
.venv-runtime/bin/python scripts/phase_2_runtime/measure.py
```

Phase 3 runs static baselines on C001–C009. Phase 4 combines the evidence into summaries:

```bash
python3 scripts/phase_3_baseline/run_static.py --bin-dir .venv-tools/bin
.venv-runtime/bin/python scripts/phase_4_analysis/summarize.py
```

## About the project

The study examines whether four existing static tools—torch-preflight, TorchFix, Semgrep, and CodeQL—detect code that keeps autograd history alive unintentionally. Each case pairs a vulnerable program with a corrected version; runtime measurements and numerical checks help confirm that the correction addresses the intended behavior.

There are ten cases, C000–C009. Phase 1 is the pilot and covers C000–C005. Phases 2–4 evaluate C001–C009; C000 is pilot-only. The detector and paper are future work, pending supervisor and peer review.

Each `vulnerable.py` and `fixed.py` is self-contained: it defines its own model, deterministic inputs, bug or repair, and a small command-line entry point. Run either variant directly, for example `python cases/case_001/vulnerable.py --device cpu --iterations 20 --seed 42`. For measured runs, the phase runner imports the same script and calls `run_experiment(device, iterations, seed, observer=...)`; the observer receives the live case state after each step. Both variants use the same interface and controls.

The case launchers run the correct study phase for that case: `PYTHON=.venv-runtime/bin/python bash cases/case_NNN/run.sh`. Phase 1 includes C000–C005; Phases 2–4 use C001–C009, with C000 reserved for the pilot.

Phase 2 runs each variant for 120 measured steps after five warm-ups, with three repetitions and seeds 42–44. CodeCarbon uses Montréal, Québec as an emissions calculation proxy. Tool versions are recorded in `results/phase_0_recon/`; its pinned dependencies are in `baselines/requirements.lock` and `baselines/codecarbon_requirements.lock`.

Each phase writes its results to the matching `results/phase_*` directory. Its scripts are in `scripts/phase_*`; common helpers are in `scripts/shared/`.
