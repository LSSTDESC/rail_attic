***************
Immediate plans
***************

An outline of the baseline RAIL is illustrated `here <https://docs.google.com/drawings/d/1or8xyBqLkpc_4_Cr-ROSA3F7fBm3RMRnRzytorw_FYM/edit?usp=sharing>`_.

1. MonoRAIL: Build the basic infrastructure for controlled experiments of forward-modeled photo-z posteriors
============================================================================================================

* a `rail.creation` submodule that can generate true photo-z posteriors and mock photometry.

* an `rail.estimation` submodule with a class for photo-z posterior estimation routines, including a template example implementing the trainZ (experimental control) algorithm.

* an `rail.evaluation.metric` submodules that calculate the metrics from the `PZ DC1 Paper <https://github.com/LSSTDESC/PZDC1paper>`_ for estimated photo-z posteriors relative to the true photo-z posteriors.

* documented scripts that demonstrate the use of RAIL in a DC1-like experiment on NERSC.

* an LSST-DESC Note presenting the RAIL infrastructure.

2. RAILroad: Quantify the impact of nonrepresentativity (imbalance and incompleteness) of a training set on estimated photo-z posteriors by multiple machine learning methods
=============================================================================================================================================================================

* a `rail.creation.degradation` submodule that introduces an imperfect prior of the form of nonrepresentativity into the observed photometry.

* at least two `rail.estimation.estimator` wrapped machine learning-based codes for estimating photo-z posteriors.

* additional `rail.evaluation.metric` modules implementing the `qp <https://github.com/LSSTDESC/qp>`_ metrics.

* documented scripts that demonstrate the use of RAIL in a blinded experiment on NERSC.

* an LSST-DESC paper presenting the results of a controlled experiment of non-representativity.

