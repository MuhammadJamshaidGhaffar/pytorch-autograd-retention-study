# C006: recurrent state graph chain

Public source: [85200](https://discuss.pytorch.org/t/convolutional-lstm-retain-graph-error/85200). Checked 2026-10-12.

Recurrent hidden/prediction history connects successive graphs. Reporter confirms detaching old predictions resolves the issue.

Provenance: **report_inspired_adaptation**. Tanh recurrence replaces ConvLSTM and old-prediction tuple; no backward/optimizer is used. Fix is valid for inference or a deliberate truncation boundary, not full backpropagation through time.

The marked case logic is in both `vulnerable.py` and `fixed.py`; the retention statement is line 12. `fixed.py` detaches at the same storage boundary. Both versions preserve numerical retained values (tested). Gradients through retained state are deliberately unnecessary for this pilot workload.

Run from the repository root:

```bash
PYTHON=.venv-runtime/bin/python bash cases/case_006/run.sh
```

Use the path relative to your repository checkout. Requires pinned CUDA PyTorch; `--device cpu` is available for semantic checks but does not supply the GPU acceptance evidence.

Results: `../../results/phase_2_runtime/case_006/` (80 iterations, 5 warm-up, seeds 42–44, fresh processes). Static output: `../../results/phase_3_baseline/static/case_006/`. Classification is runtime **CONFIRMED for this reproducer**, not confirmation that the entire original project or its historical environment was reproduced.

Detachment removes graph-associated retention. Collections can still accumulate detached tensor payload; total memory is not claimed to be constant. Do not count C000/C001/C003 as independent defects.

Run either version directly: `python cases/case_006/vulnerable.py --device cpu --iterations 20` or use `fixed.py`. The case `run.sh` launcher runs its configured measurement phase.
