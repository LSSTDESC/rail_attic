********
Overview
********

RAIL is the LSST-DESC framework for redshift assessment.
There are three aspects to the RAIL approach: creation, estimation, and evaluation. 
Each is defined by a minimal version that can be developed further as necessary.
The purpose of each piece of infrastructure is outlined below.
RAIL will eventually also comprise a fourth package: summarization.

`creation`
==========

**Creation modules**: This code enables the generation of mock photometry corresponding to a fully self-consistent forward model of the joint probability space of redshift and photometry.

**Creation base design**: We will begin with a mock data set of galaxy redshifts and photometry, for example, the DC2 extragalactic catalog.
Because any existing data set is limited in number and coverage in the 6+1 dimensional space of redshifts and photometry,
we will expand that mock data set into a continuous probability density in the space of redshifts and photometry.
This process may be done using a GAN to augment the data until it fills out the space, followed by a smoothing or interpolation of that space.
Galaxy redshifts and photometry drawn from that joint probability density will have a true likelihood and a true posterior.
This code may or may not be built off of existing tools made for the testing suite of `chippr <https://github.com/aimalz/chippr>`_.

**Creation future extensions**: In the future, we may need to consider a probability space with more data dimensions,
such as galaxy images and/or positions in order to consider codes that infer redshifts using photometric information and other sources of information.
Similarly, to evaluate template-fitting codes, we will need to construct the joint probability space of redshifts and photometry from a mock data set of SEDs and redshifts,
which could include complex effects like emission lines.

**Degradation modules**: The code in the degradation submodule enables the introduction of physical systematics into photometric training/test set pairs via the forward model of the `creation` modules.
The high-dimensional probability density outlined in the `creation` directory can be modified in ways that reflect the realistic mismatches between training and test sets.
Training and test set data will be drawn from such probability spaces with systematics applied in isolation, which preserves the existence of true likelihoods and posteriors.

**Degradation base design**: An initial experimental design would correspond to a single training set and many test sets; the systematics that can be implemented under this scheme include imbalances
between the training and test sets along the dimensions of brightness, color, and redshift.
Though it is not realistic to think of the universe in this way, realistically complex effects can still be tested in this way.
The "zeroth order" version of this infrastructure could be built using existing tools made for the testing suite of `chippr <https://github.com/aimalz/chippr>`_.

**Degradation future extensions**: An immediate extension could include the projection of measurement errors, such as a bias due to the PSF, aperture photometry parameters, or flux calibration, into the space of photometry.
Some systematics we would like to test, like incorrect values in the training set and blended galaxies, are in essence a form of model misspecification, which may be nontrivial to implement in the space of redshift
and photometry probability density and will likely not be possible with a single training set.
All effects will also need to be implemented for SED libraries in order to test template-fitting codes.

`estimation`
============

This code enables the automatic execution of arbitrary redshift estimation codes in a common computing environment.

**Motivation**: For the sake of this challenge, we will run scripts that accept test set photometry, run a particular pre-trained photo-z estimation code, and produce estimated photo-z posteriors.
Where possible, we wil use formats compatible with other LSST-DESC pipelines, including `TXPipe <https://github.com/LSSTDESC/TXPipe/>`_.
Code here will provide a script template for wrapping a machine learning code that we will run automatically on a variety of test sets blinded from those who submit scripts.
We will have to make a decision about the acceptable output format(s) of redshift posteriors.

**Structure**: Wrapped codes can be found as submodules of `rail.estimation.algos`.
Each must correspond to a config file in with any parameters the method needs, examples of which can be found in the `examples/configs` directory.

**Usage**: In the `example` directory, execute ``python main.py configs/randomPZ.yaml``.

**Immediate next steps**: `base.yaml` should not be hardcoded anywhere and should instead appear only in `main.py`.
`utils.py` is a placeholder and should be eliminated, and i/o functions should be migrated elsewhere.
There should be more examples of categories of nested config parameters in the `.yaml` files.
The `rail.estimation` module needs tests ASAP.

**Future extensions**: It may not be possible to isolate some complex `degradation` effects in a shared training set,
so future versions will require an additional script for each machine-learning-based code that executes a training step.
The estimation scripts for codes that do not naively apply machine learning to photometry, instead requiring observed information beyond photometry,
will need to accept different forms of data, so we must design the estimation framework to be flexible about input formats.
Similarly, the framework must be flexible enough to provide an SED template library and priors to template-based redshift estimation codes.

`evaluation`
============

This code is for assesing the performance of redshift estimation codes on the basis of various metrics.

**Base design**: The minimal example compares the true redshift posteriors to the estimated redshift posteriors using standard metrics of univariate probability densities, such as the Kullback-Leibler divergence.
This code may be built on the `qp <https://github.com/LSSTDESC/qp>`_ library and/or code developed for the `PZ DC1 paper <https://github.com/LSSTDESC/PZDC1paper>`_.
Estimation codes should also be compared on the basis of computational requirements.

**Future extensions**: An immediate extension would propagate estimated redshift posteriors to science-motivated metrics, such as estimators of the overall redshift distribution;
one could imagine this branch of more sophisticated metrics for DESC being called Dark Energy Redshift Assessment Infrastructure Layers (DERAIL).
