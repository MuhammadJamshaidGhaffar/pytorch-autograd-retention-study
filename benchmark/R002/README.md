# YOLOX meter after backward (R002)

Source: [https://github.com/Megvii-BaseDetection/YOLOX/issues/103](https://github.com/Megvii-BaseDetection/YOLOX/issues/103).

Exact affected classes before/after upstream fix; small training driver preserves ordinary backward before logging. Unused GPU utilities omitted.

This is a source-grounded reduction, not a full application reproduction. Persistent roots: meter deque and running total. See metadata.json for source spans and hashes, fix.patch for the paired correction, and ../../docs/phase2_protocol.md for the oracle. Run `python3 benchmark/measure.py --case R002` from the repository root.
