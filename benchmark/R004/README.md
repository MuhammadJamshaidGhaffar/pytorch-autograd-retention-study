# Repeated copy into registered buffer (R004)

Source: [https://discuss.pytorch.org/t/copy-operations-get-repeated-in-autograd-computation-graph/122547](https://discuss.pytorch.org/t/copy-operations-get-repeated-in-autograd-computation-graph/122547).

Forum minimal model dimensions, Adam and MSE preserved; graph visualization/printing omitted, repeats extended. Detach destination before copy preserves current-step parameter gradient path; DataParallel not tested.

This is a source-grounded reduction, not a full application reproduction. Persistent roots: registered tensor_bidon. See metadata.json for source spans and hashes, fix.patch for the paired correction, and ../../docs/phase2_protocol.md for the oracle. Run `python3 benchmark/measure.py --case R004` from the repository root.
