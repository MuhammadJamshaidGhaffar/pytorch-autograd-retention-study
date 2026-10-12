# Indexed diagnostic scores (R001)

Source: [https://discuss.pytorch.org/t/memory-leak-when-appending-tensors-to-a-list/25937](https://discuss.pytorch.org/t/memory-leak-when-appending-tensors-to-a-list/25937).

Report supplies score shape and indexed storage; unspecified producer replaced with small Linear and softmax. Uses dict indices instead of preallocated array. No backward as reported.

This is a source-grounded reduction, not a full application reproduction. Persistent roots: indexed scores. See metadata.json for source spans and hashes, fix.patch for the paired correction, and ../../docs/phase2_protocol.md for the oracle. Run `python3 benchmark/measure.py --case R001` from the repository root.
