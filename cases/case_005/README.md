# C005: running statistic graph chain

Public source: [16](https://discuss.pytorch.org/t/how-does-batchnorm-keeps-track-of-running-mean/40084/16). Checked 2026-10-12.

Attached running statistics are diagnosed; posts 17 and 20 confirm no_grad/detach corrections.

Provenance: **public_mechanism_reproducer**. Scalar EMA and small MLP replace complex-valued normalization; update dependency is preserved, but original training and covariance calculations are omitted.

The source is in `vulnerable.py`; the retention statement is line 13. `fixed.py` detaches at the same storage boundary. Both versions preserve numerical retained values (tested). Gradients through retained state are deliberately unnecessary for this pilot workload.

Run from any directory:

```bash
PYTHON=python3 bash cases/case_005/run.sh
```

Use the path relative to your repository checkout. Requires pinned CUDA PyTorch; `--device cpu` is available for semantic checks but does not supply the GPU acceptance evidence.

Results: `../../results/pilot_runtime/case_005/` (80 iterations, 5 warm-up, seeds 42–44, fresh processes). Static output: `../../results/pilot_static/case_005/`. Classification is runtime **CONFIRMED for this reproducer**, not confirmation that the entire original project or its historical environment was reproduced. See [pilot memo](../../docs/pilot_memo.md).

Detachment removes graph-associated retention. Collections can still accumulate detached tensor payload; total memory is not claimed to be constant. Do not count C000/C001/C003 as independent defects.
