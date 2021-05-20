# RAIL evaluation modules

The code here is for assesing the performance of redshift estimation codes on the basis of various metrics.

## Base design

The minimal example compares the true redshift posteriors to the estimated redshift posteriors using standard metrics of univariate probability densities, such as the Kullback-Leibler divergence.
This code is built based on the [`qp`](https://github.com/LSSTDESC/qp) library and inspired by the code developed for the [PZ DC1 paper](https://github.com/LSSTDESC/PZDC1paper).
Estimation codes should also be compared on the basis of computational requirements.

 The module is based on two superclasses, one to handle the input data (class Sample) and the other to handle the metrics (class Metrics). The individual metrics are instantiated as independent classes that feed the parent class via class composition. Both Sample and Metrics classes provide basic diagnostic plots and methods to return results and metadata in text format. 

<img src="UML.png" width="500"/>

The metrics are easily computed using the command line mode (evaluator.py). There is a symbolic link to this script inside the RAIL/examples directory. There is also a Jupyter Notebook available (demo.ipynb) where all functionalities are demonstrated using the toy data  available. 




## Future extensions

An immediate extension would propagate estimated redshift posteriors to science-motivated metrics, such as estimators of the overall redshift distribution; 
one could imagine this branch of more sophisticated metrics for DESC being called Dark Energy Redshift Assessment Infrastructure Layers (DERAIL).

