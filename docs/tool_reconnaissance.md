# Baseline reconnaissance

Use [results/tools_manifest.csv](../results/tools_manifest.csv) as the authority. `generate_tools_manifest.py` verifies installations without overwriting reviewed settings; `--refresh` archives the old manifest before an explicit version update. Rerun affected baselines after updates. Each execution stores argv, working directory, elapsed time, exit status, stdout and stderr; raw commands provide finer detail than the summary manifest.

| Tool | Configuration and scope | Reproducibility evidence |
|---|---|---|
| torch-preflight 0.5.0 | Default plus explicit selection of every documented TG rule; JSON; one worker | Published wheel metadata; source tag `v0.5.0`, commit `e0c44f69905a40c3560d3dafbfb48cbc8baf9a2c`; rules/explanation outputs |
| TorchFix 0.7.0 | `--select ALL -j 1`; no autofix | Wheel metadata and dependency lock; all rules includes normally disabled TOR4xx/TOR9xx families |
| CodeQL CLI 2.27.0 | Python query pack 1.8.10, existing `python-security-and-quality.qls` | Release metadata, SARIF rule catalog and extraction/analysis logs; vulnerable/fixed source trees separated |
| Semgrep 1.180.0 | Existing frozen `p/python` bundle; JSON; metrics/version check off | Exact unmodified bundle in `baselines/semgrep-python.yaml`, SHA-256 in manifest, registry version metadata in rules |

The Semgrep configuration is explicit `p/python`, not the old runner's mutable `--config auto` or an unexecuted `p/default` claim. The bundle is JSON, which is valid YAML. The engine license is LGPL-2.1-or-later; rule metadata carries the separate Semgrep Rules License. Retain that metadata when distributing the artifact. CodeQL CLI terms differ from the MIT-licensed query sources.

The principal comparator is [torch-preflight](https://github.com/highwaterlabs/torch-preflight/tree/v0.5.0). Its provenance analysis already handles assignments, arithmetic, tensor methods and some local helper returns. TG001 recognizes collection accumulation and certain persistent instance containers, with autograd/deferred-use suppressions. Its tests include self-container appends outside a lexical loop. Therefore the observed logger/helper/hook misses are specific failures to join the relevant call/context/provenance information, not evidence that it is merely a syntax linter. Ordinary attribute recurrence is also distinct from its collection/augmented-assignment candidates. These explanations are source-informed hypotheses, not exhaustive causal ablations.

[TorchFix](https://github.com/meta-pytorch/torchfix) supplies framework-specific API/best-practice linting. CodeQL and Semgrep are configurable analysis platforms. Their results here characterize the selected existing rule sets only; they do not establish that the platforms are incapable of implementing a retention analysis.

## Oracle and integrity

A relevant finding must identify the known source, sink or related flow location and explain the retention mechanism. A warning that persists on the fix is non-specific. Unused imports are unrelated even when a file contains a defect. Missing tools, failed scans, parse errors and incomplete scans are ERROR/NOT_RUN. New unadjudicated rule IDs require REVIEW; do not silently count them as clean detections or misses.

All experiment file hashes are saved in `results/pilot_static/source_hashes.json`. Source implementations were retained after seeing baseline outputs, including the detected nested-record case. Metadata/provenance was subsequently enriched. The default/all comparisons are separate rows in `baseline_results.csv`.

The overlap review inspected pinned documentation, TG001 implementation and `tests/test_rules.py`. Direct list accumulation is a known mechanism. Exact prior-example independence for other cases is UNKNOWN because issue history and every regression artifact were not exhaustively audited. An UNKNOWN field must not be converted into NO.

Raw dynamic diagnostics are separate from static scoring. Profiler operator attribution is credited; source stacks absent in this build are not. CUDA history/snapshots do provide allocation source frames, but they do not automatically establish an unintended persistent ownership relationship.

Historical SonarQube entries are archived. It was removed from the active scope at the user's request, and no unavailable SonarQube run is scored as a miss.
