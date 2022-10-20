# RAIL evaluation modules

The code here is for assessing the performance of redshift estimation codes on the basis of various metrics.

## Base design

The minimal example compares the true redshift posteriors to the estimated redshift posteriors using standard metrics of univariate probability densities, such as the Kullback-Leibler divergence.
This code is built based on the [`qp`](https://github.com/LSSTDESC/qp) library and inspired by the code developed for the [PZ DC1 paper](https://github.com/LSSTDESC/PZDC1paper).
Estimation codes should also be compared on the basis of computational requirements.




The metrics are easily computed using the command line mode (main.py). There is a symbolic link to this script inside the RAIL/examples directory. 
There is also a Jupyter Notebook available (demo.ipynb) where all functionalities are demonstrated using the toy data available. 



## Future extensions

An immediate extension would propagate estimated redshift posteriors to science-motivated metrics, such as estimators of the overall redshift distribution; 
one could imagine this branch of more sophisticated metrics for DESC being called Dark Energy Redshift Assessment Infrastructure Layers (DERAIL).

