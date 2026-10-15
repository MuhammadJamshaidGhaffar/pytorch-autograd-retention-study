# C002: cross method metric logger

Public source: [103](https://github.com/Megvii-BaseDetection/YOLOX/issues/103). Checked 2026-10-12.

YOLOX maintainers identify logging attached loss tensors among memory issues; commit d894530 adds detach before meter updates.

Provenance: **report_inspired_adaptation**. A small logger with an unbounded list replaces YOLOX MeterBuffer/AverageMeter. No optimizer or backward is executed. This is not a faithful reproduction of the original training bug.

[Verified upstream fix](https://github.com/Megvii-BaseDetection/YOLOX/commit/d89453018099062ad0a15d6df2db9a3c65fd3a1a). The logger shape was implemented before this additional provenance was located; it was not derived verbatim from the patch.

The marked case logic is in both `vulnerable.py` and `fixed.py`; the retention statement is line 11. `fixed.py` detaches at the same storage boundary. Both versions preserve numerical retained values (tested). Gradients through retained state are deliberately unnecessary for this pilot workload.

Run from the repository root:

```bash
PYTHON=.venv-runtime/bin/python bash cases/case_002/run.sh
```

Use the path relative to your repository checkout. Requires pinned CUDA PyTorch; `--device cpu` is available for semantic checks but does not supply the GPU acceptance evidence.

Results: `../../results/phase_2_runtime/case_002/` (80 iterations, 5 warm-up, seeds 42–44, fresh processes). Static output: `../../results/phase_3_baseline/static/case_002/`. Classification is runtime **CONFIRMED for this reproducer**, not confirmation that the entire original project or its historical environment was reproduced.

Detachment removes graph-associated retention. Collections can still accumulate detached tensor payload; total memory is not claimed to be constant. Do not count C000/C001/C003 as independent defects.

Run either version directly: `python cases/case_002/vulnerable.py --device cpu --iterations 20` or use `fixed.py`. The case `run.sh` launcher runs its configured measurement phase.
