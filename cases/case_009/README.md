# Five-element sliding window (C009)

Source: [https://discuss.pytorch.org/t/how-to-prevent-this-memory-leak/47581](https://discuss.pytorch.org/t/how-to-prevent-this-memory-leak/47581).

Linear dimensions reduced from 150528→1000 to 128→32. Five-item window, zeros, requires_grad and deferred backward preserved. Implements author proposed list plus stack; validates terminal gradients, not arbitrary larger application semantics.

This case is a small reduction of a public report, not a full application reproduction. Persistent roots: window of five outputs. See metadata.json for source spans and hashes, fix.patch for the paired correction. Run either version directly: `python cases/case_009/vulnerable.py --device cpu --iterations 20` or use `fixed.py`. The case `run.sh` launcher runs its configured measurement phase.
