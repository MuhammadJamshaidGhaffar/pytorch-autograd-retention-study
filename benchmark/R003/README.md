# Complex batch normalization statistics (R003)

Source: [https://discuss.pytorch.org/t/how-does-batchnorm-keeps-track-of-running-mean/40084/16](https://discuss.pytorch.org/t/how-does-batchnorm-keeps-track-of-running-mean/40084/16).

Exact two affected classes from pinned source; independent conv training driver instead of MNIST. Fix wraps only running statistic updates in no_grad as forum recommends.

This is a source-grounded reduction, not a full application reproduction. Persistent roots: running mean and covariance. See metadata.json for source spans and hashes, fix.patch for the paired correction, and ../../docs/phase2_protocol.md for the oracle. Run `python3 benchmark/measure.py --case R003` from the repository root.
