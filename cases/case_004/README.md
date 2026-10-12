# C004: forward hook feature cache

Public source: [171667](https://discuss.pytorch.org/t/pytorch-memory-leak-feature-extraction-with-forward-hook/171667). Checked 2026-10-12.

Developer reports feature-extraction/hook memory growth; discussion addresses ineffective detach and retained graphs.

Provenance: **report_inspired_adaptation**. Our hook runs with gradient tracking and persists features across batches. The report also contains no_grad, frozen parameters and hook-lifecycle issues; those are not reproduced here.

The source is in `vulnerable.py`; the retention statement is line 12. `fixed.py` detaches at the same storage boundary. Both versions preserve numerical retained values (tested). Gradients through retained state are deliberately unnecessary for this pilot workload.

Run from any directory:

```bash
PYTHON=python3 bash cases/case_004/run.sh
```

Use the path relative to your repository checkout. Requires pinned CUDA PyTorch; `--device cpu` is available for semantic checks but does not supply the GPU acceptance evidence.

Results: `../../results/pilot_runtime/case_004/` (80 iterations, 5 warm-up, seeds 42–44, fresh processes). Static output: `../../results/pilot_static/case_004/`. Classification is runtime **CONFIRMED for this reproducer**, not confirmation that the entire original project or its historical environment was reproduced. See [pilot memo](../../docs/pilot_memo.md).

Detachment removes graph-associated retention. Collections can still accumulate detached tensor payload; total memory is not claimed to be constant. Do not count C000/C001/C003 as independent defects.
