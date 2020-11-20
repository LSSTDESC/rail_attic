# RAIL estimation modules

This code enables the automatic execution of arbitrary redshift estimation codes in a common computing environment.

## Motivation

For the sake of this challenge, we will run scripts that accept test set photometry, run a particular pre-trained photo-z estimation code, and produce estimated photo-z posteriors.
Where possible, we wil use formats compatible with other LSST-DESC pipelines, including [TXPipe](https://github.com/LSSTDESC/TXPipe/).
Code here will provide a script template for wrapping a machine learning code that we will run automatically on a variety of test sets blinded from those who submit scripts.
We will have to make a decision about the acceptable output format(s) of redshift posteriors.

## Structure

Wrapped codes can be found as submodules of `rail.estimation.algos`.
Each must correspond to a config file in with any parameters the method needs, examples of which can be found in the `examples/configs` directory.
. . .

## Usage
In the `example` directory, run the following
`python main.py configs/randomPZ.yaml`

## Immediate next steps

`base.yaml` should not be hardcoded anywhere and should instead appear only in `main.py`.
`utils.py` is a placeholder and should be eliminated, and i/o functions should be migrated elsewhere.
There should be more examples of categories of nested config parameters in the `.yaml` files.
The `rail.estimation` module needs documentation and tests ASAP.

## Future extensions

It may not be possible to isolate some complex `degradation` effects in a shared training set, so future versions will require an additional script for each machine-learning-based code that executes a training step.
The estimation scripts for codes that do not naively apply machine learning to photometry, instead requiring observed information beyond photometry, will need to accept different forms of data, so we must design the estimation framework to be flexible about input formats.
Similarly, the framework must be flexible enough to provide an SED template library and priors to template-based redshift estimation codes.
