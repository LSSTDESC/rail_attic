![tests](https://github.com/LSSTDESC/BlendingToolKit/workflows/tests/badge.svg)
[![codecov](https://codecov.io/gh/LSSTDESC/RAIL/branch/master/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/RAIL)

# RAIL: Redshift Assessment Infrastructure Layers

This repo is home to a series of LSST-DESC projects aiming to quantify the impact of imperfect prior information on probabilistic redshift estimation.
RAIL differs from [PZIncomplete](https://github.com/LSSTDESC/pz_incomplete) in that it is broken into stages, each corresponding to a manageable unit of infrastructure advancement, a specific question, and a potential publication opportunity.
By pursuing the piecemeal development of RAIL, we aim to achieve the broad goals of PZIncomplete.

## Contributing

The RAIL repository uses an issue-branch-review workflow.
When you identify something that should be done, [make an issue](https://github.com/LSSTDESC/RAIL/issues/new) for it.
To contribute, isolate [an issue](https://github.com/LSSTDESC/RAIL/issues) to work on and leave a comment on the issue's discussion page to let others know you're working on it.
Then, make a branch with a name of the form `issue/#/brief-description` and do the work on the branch.
When you're ready to merge your branch into the `master` branch, [make a pull request](https://github.com/LSSTDESC/RAIL/compare) and request that other collaborators review it.
Once the changes have been approved, you can merge and squash the pull request.

## Immediate Plans

An outline of the baseline RAIL is illustrated [here](https://docs.google.com/drawings/d/1or8xyBqLkpc_4_Cr-ROSA3F7fBm3RMRnRzytorw_FYM/edit?usp=sharing).
1. _MonoRAIL_: Build the basic infrastructure for controlled experiments of forward-modeled photo-z posteriors
* a `rail.creation` submodule that can generate true photo-z posteriors and mock photometry
* an `rail.estimation` submodule with a class for photo-z posterior estimation routines, including a template example implementing the trainZ (experimental control) algorithm
* an `rail.evaluation.metric` submodules that calculate the metrics from the [PZ DC1 Paper](https://github.com/LSSTDESC/PZDC1paper) for estimated photo-z posteriors relative to the true photo-z posteriors
* documented scripts that demonstrate the use of RAIL in a DC1-like experiment on NERSC
* an LSST-DESC Note presenting the RAIL infrastructure
2. _RAILroad_: Quantify the impact of nonrepresentativity (imbalance and incompleteness) of a training set on estimated photo-z posteriors by multiple machine learning methods
* a `rail.creation.degradation` submodule that introduces an imperfect prior of the form of nonrepresentativity into the observed photometry
* at least two `rail.estimation.estimator` wrapped machine learning-based codes for estimating photo-z posteriors
* additional `rail.evaluation.metric` modules implementing the [qp](https://github.com/LSSTDESC/qp) metrics
* documented scripts that demonstrate the use of RAIL in a blinded experiment on NERSC
* an LSST-DESC paper presenting the results of a controlled experiment of non-representativity

## Future Plans

The next stages (tentative project codenames subject to change) can be executed in any order or even simultaneously and may be broken into smaller pieces each corresponding to an LSST-DESC Note.
* Extend the imperfect prior models and experimental design to accommodate template-fitting codes _(name TBD)_
* _Off the RAILs_: Investigate the effects of erroneous spectroscopic redshifts (or uncertain narrow-band photo-zs) in a training set
* _Third RAIL_: Investigate the effects of imperfect deblending on estimated photo-z posteriors
* _RAIL gauge_: Investigate the impact of measurement errors (PSF, aperture photometry, flux calibration, etc.) on estimated photo-z posteriors
* _DERAIL_: Propagate the impact of imperfect prior information to 3x2pt cosmological parameter constraints
* _RAIL line_: Implement a more sophisticated true photo-z posterior model with SEDs and emission lines
