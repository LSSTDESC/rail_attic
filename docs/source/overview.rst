********
Overview
********

RAIL (Redshift Assessment Infrastructure Layers) is the LSST-DESC framework for assessment and computation of redshifts.
Testing of multiple photo-z codes in the presence of complex systematic effects in training data has proven to be difficult due to the variety of factors that need to be controlled, e.g. differing output storage formats, assumptions inherent to individual codes, even machine architectures when run by multiple challenge participants.  RAIL seeks to minimize such impacts by unifying much of the infrastructure in a single code base.  Individual estimation codes will be "wrapped" as RAIL estimation stages so that they can be run in a controlled way.  The use of a unified `qp` Ensembles as an output format also enables a more even evaluation of the resultant PDF distributions (see `the qp repository <https://github.com/LSSTDESC/qp`_ for more details, though in brief, `qp` enables transformation between different PDF parameterizations, computation of many useful metrics, and easy fileIO).

Beyond comparison of codes, RAIL will be employed to generate photo-z catalogs used by DESC members in their science analyses. the `qp` Ensemble format is the expected default storage format for redshift information within DESC, and should be directly accesible to LSST-DESC pipelines such as `TXPipe <https://github.com/LSSTDESC/TXPipe/>`_.


There are four aspects to the RAIL approach: creation, estimation, and evaluation for individual galaxy PDFs, and summarization of ensemble redshift distributions. 
Each is defined by a minimal version that can be developed further as necessary.
The purpose of each piece of infrastructure is outlined below.

All redshift PDFs, for both individual galaxies, ensembles, and tomographic bin estimates, will be stored as `qp` Ensemble objects.

For a working example illustrating all four components of RAIL, see the `examples/goldenspike/goldenspike.ipynb <https://github.com/LSSTDESC/RAIL/blob/main/examples/goldenspike/goldenspike.ipynb>`_ jupyter notebook.

`creation`
==========

**Creation modules**: This code enables the generation of mock photometry corresponding to a fully self-consistent forward model of the joint probability space of redshift and photometry.  This forward model-based approach can provide a "true PDF" for each galaxy, enabling novel metrics for individual galaxies that are not available from catalogs where only a single true redshift is available.

**Creation base design**: We will begin with a mock data set of galaxy redshifts and photometry, for example, the DC2 extragalactic catalog.
While multiple generating engines are possible, our initial implementation utilizes a normalizing flow via the `pzflow package <https://github.com/jfcrenshaw/pzflow>`_. The normalizing flow model fits the joint distribution of redshift and photometry (and any other parameters that are supplied as inputs), and galaxy redshifts and photometry drawn from that joint probability density will have a true likelihood and a true posterior.
This 

**Creation future extensions**: In the future, we may need to consider a probability space with more data dimensions,
such as galaxy images and/or positions in order to consider codes that infer redshifts using, e.g. morphological, positional, or other sources of information.
Similarly, to evaluate template-fitting codes, we will need to construct the joint probability space of redshifts and photometry from a mock data set of SEDs and redshifts,
which could include complex effects like emission lines.
Future development of the code may or may not include incorporation of existing tools made for the testing suite of `chippr <https://github.com/aimalz/chippr>`_.

**Degradation modules**: The code in the degradation submodule enables the introduction of physical systematics into photometric training/test set pairs via the forward model of the `creation` modules.
The high-dimensional probability density outlined in the `creation` directory can be modified in ways that reflect the realistic mismatches between training and test sets, for example inclusion of photometric errors due to observing effects, spectroscopic incompleteness from specific surveys, incorrect assigment of spectroscopic redshift due to line confusion, the effects of blending, etc...
Training and test set data will be drawn from such probability spaces with systematics applied in isolation, which preserves the existence of true likelihoods and posteriors, though applying multiple degraders in series enables more complex selections to be built up to complex levels of degradation. 

**Degradation base design**: The base design for degraders in our current scheme is that degraders take in a DataFrame (or a creator that can generate samples on the fly) and returns a modified dataframe with the effects of exactly one systematic degradation.  That is, each degrader module should model one isolated degradation of the data, and more complex models are built by chaining degraders together.  While the real Universe is usually not so compartmentalized in how systematic uncertainties arise, realistically complex effects should still be testable when a series of chained degraders are applied.  RAIL has several degraders currently included: a (point-source-based) photometric error model, spectroscopic redshift LineConfusion misassignment, a simple redshift-based incompleteness, and generic QuantityCut degrader that lets the user cut on any single quantity. 

**Usage**: In the `example` directory, you can execute the creation/posterior-demo.ipynb and creation/degredation-demo.ipynb notebooks.

**Degradation future extensions**: Building up a library of degraders that can be applied to mock data in order to model the complex systematics that we will encounter is the first step of extending functionality.  Some systematics that we would like to investigate, such as incorrect values in the training set and blended galaxies, are in essence a form of model misspecification, which may be nontrivial to implement in the space of redshift and photometry probability density, and will likely not be possible with a single training set.
All effects will also need to be implemented for SED libraries in order to test template-fitting codes.
Future extensions to creation/degredation could also be built using existing tools made for the testing suite of `chippr <https://github.com/aimalz/chippr>`_.

`estimation`
============

The estimation module enables the automatic execution of arbitrary redshift estimation codes in a common computing environment.  Each photo-z method usually has both a `train` method that trains a model based on a dataset with known redshifts, and an `estimate` method that executes the particular estimation method.

**base design**: Estimators for for several popular codes `BPZ_lite` (a slimmed down version of the popular template-based BPZ code), `FlexZBoost`, and delight `Delight` are included in rail/estimation, as are an estimator `PZFlowPDF` that uses the same normalizing flow employed in the creation module, and `KNearNeighPDF` for a simple color-based nearest neighbor estimator.  The pathological `trainZ` estimator is also implemented.

**Usage**: In the `example` directory, you can execute the estimation/RAIL_estimation_demo.ipynb notebook.  Estimation codes can also be run as ceci modules with variables stored in a yaml file.

**Immediate next steps**: Adding more wrapped estimator codes so that they can be compared.

`evaluation`
============

The evalution module contains metrics for assesing the performance of redshift estimation codes.  This can be done for "true" redshift draws from a distribution or catalog, or by comparing the marginalized "true" redshift PDFs from the creation module to the estimated PDFs.

**Base design**: The starting point for the evaluation module is to include metrics employed in the PZ DC1 paper `Schmidt & Malz et al. 2020  <https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.1587S/abstract>`_. Some simple evaluation metrics will employ aspects of the `qp <https://github.com/LSSTDESC/qp>`_ codebase (e.g. computing CDF values for Probability Integral Transform, aka PIT, distributions).

**Usage**: In the `example` directory, you can execute the evaluation/demo.ipynb jupyter notebook.

**Future extensions**: Expansion of the library of available metrics.  An immediate extension would propagate estimated redshift posteriors to science-motivated metrics, and/or metrics related to computational requirements of the estimators. One could imagine this branch of more sophisticated metrics for DESC being called Dark Energy Redshift Assessment Infrastructure Layers (DERAIL).

`summarization`
===============

The summarization module houses codes that estimate redshift distributions for large ensembles of galaxies, one prominent use case being tomographic redshift bins for cosmological analyses.  Some summarizers will operate on the PDFs from the estimation stage, while others may base their redshift inference on weighted spectroscopic samples (e.g. SOM or other color-space-based schemes) or spatial clustering (e.g. the-wizz or other "clustering-z" based summarizers).  Summarizers should also have uncertainty estimates for the redshift distributions.

**Base design**:  The current summarizaation module includes very basic summarizers such as a histogram of point source estimates, the naive "stacking"/summing of PDFs, and a variational inference-based summarizer.

**Immediate next steps**: Adding more wrapped summarizer codes so that they can be compared, including at least one spatial cross-correlation method.
