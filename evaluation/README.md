# RAIL evaluation modules

The code here is for assesing the performance of redshift estimation codes on the basis of various metrics motivated by different science cases.

## Base design

The minimal example compares the true redshift posteriors to the estimated redshift posteriors using standard metrics of univariate probability densities, such as the Kullback-Leibler divergence.
This code may or may not be built on the [`qp`](https://github.com/aimalz/qp) library.

## Future extensions

An immediate extension would propagate estimated redshift posteriors to science-motivated metrics, such as estimators of the overall redshift distribution.

