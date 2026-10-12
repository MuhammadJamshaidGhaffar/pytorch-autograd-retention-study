# Comparative pilot memo — 12 October 2026

The executable pilot is complete for the user-approved four-tool scope. It establishes five misses on small GPU-validated reproducers, plus two correctly detected controls. **It does not yet satisfy the methodology's stronger requirement for five independently validated real-case reproductions.** Public reports motivate the mechanisms, but the original applications and historical training environments were not executed. Proceed to provenance-strengthening and benchmark validation; do not start the proposed detector or make a publication novelty claim yet.

## What was evaluated

The existing C000/C001 were repaired and five new scenarios, C002–C006, were added. All seven were retained in the comparison, including C003, which the strongest baseline detects. No case implementation was changed after observing its baseline outcome. The five misses below are an exploratory subset selected from those seven, not a preselected independent test set.

| Case | Persistent relation | torch-preflight default / all | TorchFix all | CodeQL security-and-quality | Semgrep frozen Python |
|---|---|---|---|---|---|
| C000 | Output → outer list | Detected / detected | Missed | Missed | Missed |
| C001 | Output → helper → list | Missed / missed | Missed | Missed | Missed |
| C002 | Scalar metric → logger method → instance list | Missed / missed | Missed | Missed | Missed |
| C003 | Output → dictionary record → list | Detected / detected | Missed | Missed | Missed |
| C004 | Activation → registered hook → feature cache | Missed / missed | Missed | Missed | Missed |
| C005 | New statistic + old statistic → persistent EMA | Missed / missed | Missed | Missed | Missed |
| C006 | New features + previous hidden state → next hidden state | Missed / missed | Missed | Missed | Missed |

“Missed” means the completed scan produced no relevant localized retention finding on this confirmed reproducer. It does not mean a tool crashed, lacks every applicable feature, or would miss all real implementations of the mechanism. CodeQL reports unused imports on six of seven cases, on both variants; these are not retention detections. No relevant retention warning remains on any fixed counterpart. Default and explicit-all torch-preflight outcomes agree.

Versions and configurations are authoritative in [tools_manifest.csv](../results/tools_manifest.csv): torch-preflight 0.5.0, TorchFix 0.7.0, CodeQL 2.27.0 with Python queries 1.8.10, and Semgrep 1.180.0 with frozen existing `p/python` rules. Semgrep changed from the historical manifest's 1.177.0; the earlier manifest is archived. SonarQube was explicitly removed by the user on 12 October. No custom competitor rules or competitor source modifications were used.

## Ground truth and experimental repairs

The prior collector incremented an iteration counter on multiple Python line events, analyzed pairs in one process, preallocated all inputs, and assigned CONFIRMED labels before applying a measured oracle. Some runner paths converted execution or SARIF parsing failures into “clean.” Metadata also had duplicate IDs, incorrect source lines, and inconsistent environment versions. The obsolete legacy archive was subsequently removed at the user's request; it is not used in current conclusions.

Each current variant/repetition runs in a fresh Python process on an RTX 5090, Python 3.12.3, PyTorch 2.14.0+cu130, CUDA runtime 13.0. The model is a 128→512→128 MLP, batch size 32. Pairs share model initialization, shapes, number of steps and seeds. There are five warm-up steps followed by 80 measured steps, repeated for seeds 42, 43 and 44. Inputs are generated lazily. One observation occurs after each completed iteration. No optimizer or backward is used: the workload needs numerical telemetry/features or recurrent inference state, not gradients through retained history.

The trace records allocated/reserved CUDA bytes, current host RSS, retained tensors, attached roots, reachable graph nodes, distinct retained output storage, and reachable saved activation storage. Model parameter and output payload storage are excluded from the saved-activation count. Private `_saved_*` introspection is version-specific corroborating evidence, not an exact accounting of all autograd memory. CUDA allocation and release after clearing persistent state provide additional checks. Profiler runs are separate because profiling can affect object lifetimes.

The operational pilot oracle checks growing graph reachability, positive retained saved activations, no graph nodes/saved activations in fixed controls, and greater measured vulnerable allocation growth and cleanup release. Its 1 MiB paired-growth threshold was adopted during this exploratory pilot, not preregistered; it must not be presented as a universal leak threshold. All three repetitions satisfy it for every case.

| Case | Vulnerable growth, MiB | Fixed growth, MiB | Paired excess, MiB |
|---|---:|---:|---:|
| C000 | 7.4062 | 1.2344 | 6.1719 |
| C001 | 7.4062 | 1.2344 | 6.1719 |
| C002 | 7.4448 | 0.0386 | 7.4062 |
| C003 | 7.4062 | 1.2344 | 6.1719 |
| C004 | 6.1719 | 4.9375 | 1.2344 |
| C005 | 6.1719 | 0.0000 | 6.1719 |
| C006 | 7.4062 | 0.0000 | 7.4062 |

Growth is final minus first measured allocation, averaged across repetitions. The traces agree across seeds at the displayed precision. Detached output lists still grow in payload; **only graph-associated accumulation is removed**. C005 and C006 retain constant-size tensor state while their vulnerable graph histories grow; their fixed allocation remains bounded. Tests also verify numerical equivalence of retained values, overwrite/cleanup behavior, no-grad payload growth, and release of saved activations after backward despite surviving graph nodes.

## Public provenance and what it supports

All links and transformation notes are in [candidate_cases.csv](../results/candidate_cases.csv) and individual case READMEs.

C000 follows the [public output-list report](https://discuss.pytorch.org/t/memory-leak-when-appending-tensors-to-a-list/25937). C001 and C003 deliberately transform that same mechanism; they are not independent defects. C002 is corroborated by [YOLOX issue 103](https://github.com/Megvii-BaseDetection/YOLOX/issues/103) and [the meter detachment fix](https://github.com/Megvii-BaseDetection/YOLOX/commit/d89453018099062ad0a15d6df2db9a3c65fd3a1a), but substitutes a simpler accumulating logger and omits training. The GitHub provenance was located after that small logger was implemented; it does not retroactively make the implementation an original-project reduction.

C004 is motivated by the [forward-hook feature-extraction discussion](https://discuss.pytorch.org/t/pytorch-memory-leak-feature-extraction-with-forward-hook/171667). The original thread includes frozen parameters, no-grad regions and hook-lifecycle complications; our attached persistent-cache mechanism isolates only one risk. C005 preserves the dependency in the [confirmed running-statistics correction](https://discuss.pytorch.org/t/how-does-batchnorm-keeps-track-of-running-mean/40084/16), replacing complex normalization with a scalar EMA. C006 simplifies the [ConvLSTM history discussion](https://discuss.pytorch.org/t/convolutional-lstm-retain-graph-error/85200). Detaching recurrent history changes training gradients, so our correction is justified only for the declared inference/truncation intent.

Known-to-tool status is YES at the mechanism level for the direct-list control. The remaining statuses are UNKNOWN, not “independent.” The pinned torch-preflight source has self-container and helper-return regression tests, so it would be incorrect to claim that it has no interprocedural reasoning. See [overlap review](../results/baseline_overlap.csv).

## Runtime diagnostics

The PDF batch/request adaptations and their results were removed from the active study at the user's request. They do not contribute to any reported case count.


PyTorch Profiler and CUDA history/snapshot experiments ran on all seven pairs. Profiler output exposes allocating operators. In this build, profiler events contain no source stacks despite `with_stack=True`; source localization is therefore not credited to the profiler. CUDA snapshots contain case-file allocation frames for every vulnerable case, and show retained live allocations. Neither tool automatically explains the complete producer-to-persistent-owner relationship in these runs; that interpretation remains manual. They are complementary diagnostics, not entries in the static recall table. Raw diagnostic artifacts remain local and hashed; compressed and uncompressed snapshots/profiler traces are ignored by Git. Small summaries and regeneration code remain versioned.

## Gate recommendation

There is enough repeatable evidence to investigate cross-call persistent storage and recursive state further. There is not yet enough independent real-case evidence to approve a new detector. First preserve original-source versions and reduce actual failures without changing their autograd use, obtain at least two independent cases for a coherent gap, and complete baseline overlap review. Then have the supervisor decide whether the full benchmark and gap support enhanced analysis. The detector, held-out evaluation, natural-repository study and paper remain later phases.

This memo records Phase 1 only. Subsequent source-grounded reproductions and Gate 2 evidence are in [phase4_review.md](phase4_review.md).
