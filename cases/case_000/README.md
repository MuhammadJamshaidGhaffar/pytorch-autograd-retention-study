# C000: direct output list

Public source: [25937](https://discuss.pytorch.org/t/memory-leak-when-appending-tensors-to-a-list/25937). Checked 2026-10-12.

Forum describes attached outputs stored in a list and recommends detachment.

Provenance: **public_mechanism_reproducer**. Deterministic small MLP, fixed workload and explicit measurement replace the original model.

The source is in `vulnerable.py`; the retention statement is line 12. `fixed.py` detaches at the same storage boundary. Both versions preserve numerical retained values (tested). Gradients through retained state are deliberately unnecessary for this pilot workload.

Run from any directory:

```bash
PYTHON=python3 bash cases/case_000/run.sh
```

Use the path relative to your repository checkout. Requires pinned CUDA PyTorch; `--device cpu` is available for semantic checks but does not supply the GPU acceptance evidence.

Results: `../../results/pilot_runtime/case_000/` (80 iterations, 5 warm-up, seeds 42–44, fresh processes). Static output: `../../results/pilot_static/case_000/`. Classification is runtime **CONFIRMED for this reproducer**, not confirmation that the entire original project or its historical environment was reproduced. See [pilot memo](../../docs/pilot_memo.md).

Detachment removes graph-associated retention. Collections can still accumulate detached tensor payload; total memory is not claimed to be constant. Do not count C000/C001/C003 as independent defects.
