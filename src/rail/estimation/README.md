# RAIL estimation modules

This code enables the execution of arbitrary redshift characterization codes with a shared API.
It includes classes for numerous algorithms for characterizing per-galaxy photo-z PDFs and ensemble galaxy sample redshift distributions.

## Structure

Wrapped codes can be found as submodules of `rail.estimation.algos`.
Each must correspond to a config file in with any parameters the method needs, examples of which can be found in the `examples/configs` directory.
. . .

## Usage
In the `example` directory, run the following
`python main.py configs/randomPZ.yaml`

## Future extensions

To sppplement the classes of stages defined in `estimation.py` and `summaization.py`, a `classification.py` module may be added to include algorithms for defining subsamples of a galaxy sample from their photometric data and/or photo-z data products, e.g. tomographic binning procedures.

It may not be possible to isolate some complex `degradation` effects in a shared training set, so future versions will require an additional script for each machine-learning-based code that executes a training step.
The estimation scripts for codes that do not naively apply machine learning to photometry, instead requiring observed information beyond photometry such as positions or imaging, will need to accept different forms of data, so we must design the estimation framework to be flexible about input formats.
