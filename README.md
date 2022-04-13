![tests](https://github.com/LSSTDESC/BlendingToolKit/workflows/tests/badge.svg)
[![codecov](https://codecov.io/gh/LSSTDESC/RAIL/branch/master/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/RAIL)

# RAIL: Redshift Assessment Infrastructure Layers

RAIL's purpose is to be the infrastructure enabling the PZ WG Deliverables in [the LSST-DESC Science Roadmap (see Sec. 5.18)](https://lsstdesc.org/assets/pdf/docs/DESC_SRM_latest.pdf), aiming to guide the selection and implementation of redshift estimators in DESC pipelines.
RAIL differs from previous plans for PZ pipeline infrastructure in that it is broken into stages, each corresponding to a manageable unit of infrastructure advancement, a specific question to answer with that code, and a guaranteed publication opportunity.
RAIL uses [qp](https://github.com/LSSTDESC/qp) as a back-end for handling univariate probability density functions (PDFs) such as photo-z posteriors or n(z) samples.

## Installation

Installation instructions are available on the [RAIL Read The Docs page](https://lsstdescrail.readthedocs.io/en/latest/)

## Contributing

If interested in contributing to `RAIL` see the [Contributing section](https://lsstdescrail.readthedocs.io/en/latest/source/contributing.html) of the RAIL Read The Docs page.

## Immediate Plans

An outline of the baseline RAIL is illustrated [here](https://docs.google.com/drawings/d/1or8xyBqLkpc_4_Cr-ROSA3F7fBm3RMRnRzytorw_FYM/edit?usp=sharing).
1. _Golden Spike_: Build the basic infrastructure for controlled experiments of forward-modeled photo-z posteriors
- [X] a `rail.creation` subpackage that can generate true photo-z posteriors and mock photometry
- [X] an `rail.estimation` subpackage with a superclass for photo-z posterior estimation routines and at least one subclass template example implementing the trainZ (experimental control) algorithm
- [X] a `rail.evaluation` subpackage that calculates at least the metrics from the [PZ DC1 Paper](https://github.com/LSSTDESC/PZDC1paper) for estimated photo-z posteriors relative to the true photo-z posteriors
- [ ] documented scripts that demonstrate the use of RAIL in a DC1-like experiment on NERSC
- [ ] sufficient documentation for a v1.0 release
- [ ] an LSST-DESC Note presenting the RAIL infrastructure
2. _RAILroad_: Quantify the impact of nonrepresentativity (imbalance and incompleteness) of a training set on estimated photo-z posteriors by multiple machine learning methods
- [ ] parameter specifications for degrading an existing `Creator` to make an imperfect prior of the form of nonrepresentativity into the observed photometry
- [X] at least two `Estimator` wrapped machine learning-based codes for estimating photo-z posteriors
- [ ] additional `Evaluator` metrics with feed-through access to the [qp](https://github.com/LSSTDESC/qp) metrics
- [ ] end-to-end documented scripts that demonstrate a blinded experiment on NERSC
- [ ] an LSST-DESC paper presenting the results of the experiment

## Future Plans

Potential extensions of the RAIL package are summarized in the [Future Plans section](https://lsstdescrail.readthedocs.io/en/latest/source/futureplans.html) of the RAIL Read The Docs page.

##Citing RAIL

This code, while public on GitHub, has not yet been released by DESC and is still under active development. Our release of v1.0 will be accompanied by a journal paper describing the development and validation of RAIL.

If you make use of the ideas or software here, please cite this repository https://github.com/LSSTDESC/RAIL. You are welcome to re-use the code, which is open source and available under terms consistent with our [LICENSE](https://github.com/LSSTDESC/RAIL/blob/main/LICENSE) [(BSD 3-Clause)](https://opensource.org/licenses/BSD-3-Clause).

External contributors and DESC members wishing to use RAIL for non-DESC projects should consult with the Photometric Redshifts (PZ) Working Group conveners, ideally before the work has started, but definitely before any publication or posting of the work to the arXiv.
