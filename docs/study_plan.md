# Assignment interpretation and phase plan

Read: `PyTorch_Autograd_Retention_Method.docx`, `PyTorch_Autograd_Retention_Student_Tutorial.docx`, and `Unreleased_Tensor_Reference_Experiment_Guide.pdf` from `../professor_material/`. They serve different purposes. The September methodology controls the research sequence; the tutorial supplies learning exercises; the PDF supplies broader TensorFlow/Keras injection ideas.

The assigned question is whether current PyTorch-aware static tools miss independently validated autograd-retention defects, and whether a coherent extension can improve detection. The methodology does **not** require manufacturing 20 misses. It calls for a five-case comparative pilot, then approximately 8–12 strong confirmed mechanisms. Twenty instances could include variants later, but variants must not be counted as independent mechanisms.

## What to learn and demonstrate now

Understand graph reachability, `requires_grad`, `grad_fn`, detached storage, `no_grad`, `inference_mode`, and why `eval()` alone does not disable autograd. Distinguish allocated GPU memory from allocator reservations. A detached tensor can still occupy memory. An overwritten attribute can retain one graph without unbounded accumulation; a recursively updated single attribute can retain a growing chain. Ordinary backward can free saved activations while Python references keep graph nodes/output tensors alive.

Practice reading the small paired programs and their traces before designing static rules. The current tests exercise these distinctions and verify that each fix preserves numerical values under the declared workload. They do not establish readiness for interprocedural CodeQL implementation or a completed L0–L3 capstone.

The learning tutorial introduces L0 syntax, L1 flow, L2 autograd barriers, and L3 lifetime reasoning. The revised methodology explicitly postpones building those detectors until the baseline and real-gap gates. L0–L3 are internal ablations, not proof that external tools lack those capabilities.

## Work packages and acceptance

| Work package | Input and procedure | Output/schema | Acceptance | Stop/failure condition |
|---|---|---|---|---|
| WP0 tool reconnaissance | Inspect published rules and installed versions; freeze unmodified configurations | `tools_manifest.csv`, pinned dependencies/rules, command wrappers | Exact version, release/source, license and configuration; machine-readable output | Tool errors or version mismatch are ERROR, never a miss |
| WP1 exploratory pilot | Run paired reproducible cases, baseline scans, profiler/snapshots; adjudicate locations and mechanism | `pilot_results.csv`, `ground_truth.csv`, `baseline_results.csv`, raw evidence, memo | Repeated graph growth removed by correction; successful scans and relevant localization | Cannot reproduce → repair ground truth; most strong real cases solved → reconsider detector |
| WP2–WP4 independent benchmark | Recover original public source/commit and fix; preserve autograd/backward behavior when reducing | `candidate_cases.csv`, versioned pairs, source spans, traces; CONFIRMED/BOUNDARY/REJECTED | Roughly 8–12 strong mechanisms; provenance and intent established | Generic storage, capacity demand, or unresolved cause stays boundary/rejected |
| WP5–WP7 full baseline and gap study | Same four frozen baselines on all confirmed pairs; inspect competitor tests/docs/issues | `baseline_results.csv`, `baseline_overlap.csv`, predefined gap dimensions | Multiple independent real defects support a coherent missing abstraction | One-off or already solved cases do not justify an analyzer |
| Gate 2 | Supervisor reviews original-case evidence and technical gap | Written proceed/reformulate/stop decision | Contribution exceeds torch-preflight | No coherent gap → no new detector |
| WP8–WP10 conditional analyzer | Use CodeQL or another established flow framework; development cases only; freeze before held-out run | L0–L3, regression tests, held-out results | Paired improvement over torch-preflight with controlled false positives | Never tune on held-out results; document a new evaluation if changes are needed |
| WP11–WP14 natural study | Freeze selection rules and 50–100 repositories; scan both tools; validate findings | Repository snapshots, TP/candidate/FP/uncertain labels, runtime confirmations | Independently confirm high-value findings and report scalability | Unreproduced findings remain static candidates |
| WP15 paper and artifact | Analyze mechanism/project-level results, limitations, reproducibility | Tables, figures, artifact package, paper | Claims follow evidence, including negative results | Do not count variants as independent samples or claim novelty from generic-linter misses |

The user narrowed the active baseline set to **torch-preflight, TorchFix, CodeQL and Semgrep** on 12 October 2026. SonarQube remains only in historical material. This is a documented deviation from the professor's original five-tool matrix.

## Current handoff

The executable pilot and four-tool comparison are available. Five additional scenarios were implemented; two original scenarios were repaired; the PDF adaptations were subsequently removed at the user's request. Five misses were observed across the seven main reproducers. All sources and transformations are recorded, but **five independent original-project failures have not been established**. The strict real-case gate remains open. The next phase is original-source reproduction and stronger provenance, not detector construction.

C005 has particularly direct developer-confirmed mechanism/fix evidence. C002 has an upstream fix but the actual YOLOX meter/backward behavior should be preserved in a follow-up reduction. C004 and C006 need careful separation of hook lifecycle, frozen parameters, retained backward graphs and intended recurrence. C001/C003 remain transformations of C000 and cannot fill an independent-case quota.

No publication timeline or venue acceptance can be inferred from this pilot. The methodology's one-week pilot and tutorial's ten-day roadmap are planning guidance; agree subsequent deadlines with the supervisor using the actual acceptance gates above.

## Phase 2–4 handoff

See [phase4_review.md](phase4_review.md) and the frozen [protocol](phase2_protocol.md). These phases add five source-grounded reductions and evaluate their paired corrections. They do not establish full-application reproducibility, novelty, or independent human confirmation. Phase 5 is explicitly held for supervisor and peer confirmation.

The user confirmed that 20 cases was an interpretation, not a separate supervisor requirement. Follow the method's approximate 8–12 mechanism target, with fewer strong cases permitted. The five-case source benchmark currently covers four mechanism families; broader benchmark completion has not been established.
