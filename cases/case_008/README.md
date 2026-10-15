# Repeated copy into registered buffer (C008)

Source: [https://discuss.pytorch.org/t/copy-operations-get-repeated-in-autograd-computation-graph/122547](https://discuss.pytorch.org/t/copy-operations-get-repeated-in-autograd-computation-graph/122547).

Forum minimal model dimensions, Adam and MSE preserved; graph visualization/printing omitted, repeats extended. Detach destination before copy preserves current-step parameter gradient path; DataParallel not tested.

This case is a small reduction of a public report, not a full application reproduction. Persistent roots: registered tensor_bidon. See metadata.json for source spans and hashes, fix.patch for the paired correction. Run either version directly: `python cases/case_008/vulnerable.py --device cpu --iterations 20` or use `fixed.py`. The case `run.sh` launcher runs its configured measurement phase.
