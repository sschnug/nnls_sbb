# nnls_sbb
First-order non-monotonic sparse nonnegative least-squares optimization

This is basically a prototype-implementation of:

> Kim, Dongmin, Suvrit Sra, and Inderjit S. Dhillon. "A non-monotonic method for large-scale nonnegative least squares." (2010).

The implementation itself is currently unable to beat [L-BFGS-B](http://users.iems.northwestern.edu/~nocedal/lbfgsb.html) for medium-size datasets, mostly due to the kind of usage of scipy's sparse-matrices.

The test-data needed to run these benchmarks are the ones from [libsvmtools/datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).
