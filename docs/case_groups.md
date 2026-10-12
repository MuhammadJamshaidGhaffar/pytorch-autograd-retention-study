# `case_00*` versus `R00*`

These are two groups of experiments in the same project—not two detectors.

| | `cases/case_000`–`case_006` | `benchmark/R001`–`R005` |
|---|---|---|
| Purpose | **Pilot:** explore retention patterns and check the experiment setup | **Source benchmark:** reproduce behavior more closely tied to public reports or source code |
| Origin | Small examples; some are synthetic variations or simplified adaptations | Public GitHub/PyTorch reports, with source links and documented reductions |
| Execution | Simplified workloads without backward | Case-specific workloads; some preserve training and backward |
| Count | 7 scenarios | 5 source cases, grouped into 4 mechanism families |
| Main results | `results/pilot_results.csv` | `results/benchmark_results.csv` |

**Every case in both groups has `vulnerable.py` and `fixed.py`.** Those are two versions of one case, not two separate cases.

## A concrete example

`cases/case_002` uses a simple metric logger that stores attached tensors. It explores whether tools notice storage across method calls.

`benchmark/R002` uses the actual affected YOLOX meter classes and the upstream correction, with a small training driver that runs backward before logging. It provides stronger evidence about that public-source problem, while still being a reduction rather than the complete YOLOX application.

