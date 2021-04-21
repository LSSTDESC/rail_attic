![tests](https://github.com/LSSTDESC/BlendingToolKit/workflows/tests/badge.svg)
[![codecov](https://codecov.io/gh/LSSTDESC/RAIL/branch/master/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/RAIL)

# RAIL: Redshift Assessment Infrastructure Layers

This repo is home to the back-end of the LSST-DESC PZ WG's software pipelines aiming to guide the selection and implementation of redshift estimators, encompassing the former [PZIncomplete](https://github.com/LSSTDESC/pz_incomplete) and [PZCalibrate](https://github.com/LSSTDESC/pz_calibrate) codes.
RAIL differs from PZIncomplete in that it is broken into stages, each corresponding to a manageable unit of infrastructure advancement, a specific question, and a guaranteed publication opportunity.
As PZCalibrate was indeed completed, it will be wrapped for incorporation into RAIL.
RAIL uses [qp](https://github.com/LSSTDESC/qp) as a back-end for handling univariate probability density functions (PDFs) such as photo-z posteriors or n(z) samples.

## Contributing

The RAIL repository uses an issue-branch-review workflow.
When you identify something that should be done, [make an issue](https://github.com/LSSTDESC/RAIL/issues/new) for it.
To contribute, isolate [an issue](https://github.com/LSSTDESC/RAIL/issues) to work on and leave a comment on the issue's discussion page to let others know you're working on it.
Then, make a branch with a name of the form `issue/#/brief-description` and do the work on the branch.
When you're ready to merge your branch into the `master` branch, [make a pull request](https://github.com/LSSTDESC/RAIL/compare) and request that other collaborators review it.
Once the changes have been approved, you can merge and squash the pull request.

## Immediate Plans

An outline of the baseline RAIL is illustrated [here](https://docs.google.com/drawings/d/1or8xyBqLkpc_4_Cr-ROSA3F7fBm3RMRnRzytorw_FYM/edit?usp=sharing).
1. _Golden Spike_: Build the basic infrastructure for controlled experiments of forward-modeled photo-z posteriors
* a `rail.creation` submodule that can generate true photo-z posteriors and mock photometry
* an `rail.estimation` submodule with a class for photo-z posterior estimation routines, including a template example implementing the trainZ (experimental control) algorithm
* an `rail.evaluation.metric` submodules that calculate the metrics from the [PZ DC1 Paper](https://github.com/LSSTDESC/PZDC1paper) for estimated photo-z posteriors relative to the true photo-z posteriors
* documented scripts that demonstrate the use of RAIL in a DC1-like experiment on NERSC
* an LSST-DESC Note presenting the RAIL infrastructure
2. _monoRAIL_: Quantify the impact of nonrepresentativity (imbalance and incompleteness) of a training set on estimated photo-z posteriors by multiple machine learning methods
* a `rail.creation.degradation` submodule that introduces an imperfect prior of the form of nonrepresentativity into the observed photometry
* at least two `rail.estimation.estimator` wrapped machine learning-based codes for estimating photo-z posteriors
* additional `rail.evaluation.metric` modules implementing the [qp](https://github.com/LSSTDESC/qp) metrics
* documented scripts that demonstrate the use of RAIL in a blinded experiment on NERSC
* an LSST-DESC paper presenting the results of a controlled experiment of non-representativity

## Future Plans

RAIL's purpose is to be the infrastructure enabling the PZ WG Deliverables in [the LSST-DESC Science Roadmap (see Sec. 5.18)](https://lsstdesc.org/assets/pdf/docs/DESC_SRM_latest.pdf), which aim to break up the PZ WG's pipeline responsibilities into smaller milestones that can be accomplished by individuals or small groups on short timescales, i.e. under a year.
The next stages  of RAIL development (tentative project codenames subject to change) are intended to be paper-able projects each of which addresses one or more SRM Deliverables by incrementally advancing the code along the way to project completion.
They are scoped such that any can be executed in any order or even simultaneously.
* _RAILyard_: Extend the imperfect prior models and experimental design to accommodate template-fitting codes
* _Off the RAILs_: Investigate the effects of erroneous spectroscopic redshifts (or uncertain narrow-band photo-zs) in a training set
* _Third RAIL_: Investigate the effects of imperfect deblending on estimated photo-z posteriors
* _RAIL gauge_: Investigate the impact of measurement errors (PSF, aperture photometry, flux calibration, etc.) on estimated photo-z posteriors
* _DERAIL_: Propagate the impact of imperfect prior information to 3x2pt cosmological parameter constraints
* _RAIL line_: Implement a more sophisticated true photo-z posterior model with SEDs and emission lines

Informal library of fun train-themed names for future projects/pipelines built with RAIL: _RAILroad_, _tRAILblazer_
