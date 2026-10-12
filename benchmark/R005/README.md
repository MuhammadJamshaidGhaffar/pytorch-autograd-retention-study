# Five-element sliding window (R005)

Source: [https://discuss.pytorch.org/t/how-to-prevent-this-memory-leak/47581](https://discuss.pytorch.org/t/how-to-prevent-this-memory-leak/47581).

Linear dimensions reduced from 150528→1000 to 128→32. Five-item window, zeros, requires_grad and deferred backward preserved. Implements author proposed list plus stack; validates terminal gradients, not arbitrary larger application semantics.

This is a source-grounded reduction, not a full application reproduction. Persistent roots: window of five outputs. See metadata.json for source spans and hashes, fix.patch for the paired correction, and ../../docs/phase2_protocol.md for the oracle. Run `python3 benchmark/measure.py --case R005` from the repository root.
