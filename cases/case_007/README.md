# YOLOX meter after backward (C007)

Source: [https://github.com/Megvii-BaseDetection/YOLOX/issues/103](https://github.com/Megvii-BaseDetection/YOLOX/issues/103).

Exact affected classes before/after upstream fix; small training driver preserves ordinary backward before logging. Unused GPU utilities omitted.

This case is a small reduction of a public report, not a full application reproduction. Persistent roots: meter deque and running total. See metadata.json for source spans and hashes, fix.patch for the paired correction. Run either version directly: `python cases/case_007/vulnerable.py --device cpu --iterations 20` or use `fixed.py`. The case `run.sh` launcher runs its configured measurement phase.
