# C003: nested output records

Public source: [25937](https://discuss.pytorch.org/t/memory-leak-when-appending-tensors-to-a-list/25937). Checked 2026-10-12.

Same public list-retention mechanism as C000.

Provenance: **synthetic_variant**. Wrapping predictions in dictionaries is an experimental transformation; it is not a separate developer-reported defect.

The marked case logic is in both `vulnerable.py` and `fixed.py`; the retention statement is line 17. `fixed.py` detaches at the same storage boundary. Both versions preserve numerical retained values (tested). Gradients through retained state are deliberately unnecessary for this pilot workload.

Run from the repository root:

```bash
PYTHON=.venv-runtime/bin/python bash cases/case_003/run.sh
```

Use the path relative to your repository checkout. Requires pinned CUDA PyTorch; `--device cpu` is available for semantic checks but does not supply the GPU acceptance evidence.

Results: `../../results/phase_2_runtime/case_003/` (80 iterations, 5 warm-up, seeds 42–44, fresh processes). Static output: `../../results/phase_3_baseline/static/case_003/`. Classification is runtime **CONFIRMED for this reproducer**, not confirmation that the entire original project or its historical environment was reproduced.

Detachment removes graph-associated retention. Collections can still accumulate detached tensor payload; total memory is not claimed to be constant. Do not count C000/C001/C003 as independent defects.

Run either version directly: `python cases/case_003/vulnerable.py --device cpu --iterations 20` or use `fixed.py`. The case `run.sh` launcher runs its configured measurement phase.
