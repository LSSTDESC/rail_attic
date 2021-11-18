# Golden Spike Example Script.

"Golden spike" is meant to demonstrate a full "end-to-end" running of the RAIL infrastructure, something you might do to compare the performance of photo-z codes a la the PhotoZDC1 challenge described in [Schmidt et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.1587S/abstract).  There are three main modules that will be exercised within golden spike: mock data will be created with the Creation module, codes will be run with the Estimation module, and metrics computed with the Evaluation module.

You must have RAIL and all dependencies installed locally in order to run goldenspike.  Follow the instructions on the main RAIL page in order to install `creation`, `estimation`, and `evaluation` RAIL packages, i.e. run `pip install .[all]` (Note: if you are using zsh, e.g. on newer macs, then you will need to escape anything in square brackets, and will need to do `pip install '.[all]'`).  In addition, if you will need either a pre-trained pzflow object or a local set of data with which to train the RAIL/creation creator that will create your mock data.  Note that there are both an example flow (`pretrained_flow.pkl`) and a small test set of data (`test_flow_data.pq`) available in the `RAIL/examples/goldenspike/data` subdirectory.

The main script is `goldenspike.py` which reads is controlled by the many options in three yaml files, one for each major stage: `creation_goldspike.yaml`, `estimation_goldspike.yaml`, and `evaluation_goldspike.yaml`. Note that the first parameter listed in each yaml file controls whether or not that stage is run.  That is, there are three parameters, one each per yaml file, `run_creation, `run_estimation`, and `run_evaluation`, if set to True, the corresponding section of the script is run.  If set to False, then the script will look for intermediate files created in a previous run of the script.  This enables users to run e.g. only the `estimation` stage if data from a previous run of `creation` is already on disk, or just the  `evaluation` stage if `estimation` was run previously.  

Below we will go through some of the features of each stage:
**NOTE**: could switch to grab these on command line rather than hardcoded names, ask others for preference.

## Running the script
After you have adjusted all parameters in the three yaml files, run from the command line with `python goldenspike.py`.  For some details on how to set up the parameters, continue reading for a description of the options available in the yaml files that control the script.


## Creation
The major functionality of the creation module is to create mock data with which to test the set of estimators, which also produces "true" redshift posteriors for each mock galaxy.  There are two options for generating mock data: 1) starting with a pre-trained flow; and 2) loading photometry+redshift data from file.  

### Data used as input to RAIL Creator
In `creation_goldspike.yaml` setting `has_flow: True` will read in the pre-trained flow from the pickle file specified with `flow_file`.  There is an existing pretrained flow that has been trained on a subset of data from True data from healpix pixel 9816 of cosmoDC2 in the data directory named `data/pretrained_flow.pkl` that can be used in this demo.

If you instead wish to read in data from file, setting `use_local_data: True will read in the photometry and redshift info from the file specified in `local_flow_data_file` (there is an example data file named `data/test_flow_data.pq` that will work as a default.  Note that you must set `flow_columns` to an array that includes only the columns that you wish to input into the flow, as extra columns may lead to lower quality fits, as the neural flow will spend time and coefficients trying to emulate the larger space that includes the extra columns.

Once the flow is created, we must then add magnitude errors for our bands.  We will use a RAIL degrader, `LSSTErrorModel` to do this.  Parameters to control the behavior or LSSTErrorModel are specified in the yaml file under `LSSTErrorModel_params`.  `ndFlag` tells the code which values to set for objects with negative observed flux (default is NaN, but we will set to 99.0).  `highSNR` is a boolean flag that determines whether or not to use the "high S/N approximation" for magnitude errors, having this flag set to `True` will cause very large errors at S/N<~5, which can lead to problematic behavior for faint objects, so we will set the flag to `False`.  `bandNames` defines a mapping between the internal names of the bands and the names which we wish to use.  `nvisYr`, `gamma`, `Cm`, `msky`, `theta`, and `km` are model parameters taken from Ivezic et al (https://arxiv.org/abs/0805.2366) that define the model, see that paper for more detail.

In addition to our magnitude error degrader, you can choose whether to apply additional degraders with the `use_degrader` option. If it is set to False then the same creator is used for both test and training data.  If set to `True`, then the code will look for a dictionary in the yaml file called `degraders`.  The key for each `degraders` dictonary entry should be the name of the specific degrader, and any keyword arguments needed for the degrader should be stored in a dictionary for each specific degrader.  If you specify multiple degraders, the code will loop through each and apply then consecutively in the order specified in the yaml file.

The script then creates training data (using the undegraded creator) and test data (using the degraded creator, if applied), You set the number of mock galaxies created with the `N_test_gals` and `N_train_gals` parameters in the yaml file.
The "true posteriors" for each galaxy are evaluated on a grid, you set this grid with `zmin`, `zmax`, and `nzbins` in the yaml file, and set the name used for the redshift column using `z_column`.

If you wish to save the true posteriors to file, set `save_ensemble: True` and they will be written as a qp Ensemble to the filename specified in `ensemble_file`.  Similarly, if you wish to save the flow that you trained, `save_flow: True` will save the flow to the filename specified in `saved_flow_file`

The test and train datasets that were sampled from the flow and will be input to the RAIL estimators are written to the files specified by the values set in `test_filename` and `train_filename`

## Estimation
Estimation is where the actual photo-z estimation takes place, and the options in `estimation_goldspike.yaml` control which codes are run.
`base_yaml` specifies the location of a secondary yaml file specifying path and format information
```
**NOTE**: do we still need a base.yaml?  Does it change form to be separate from `estimation` because some things that we might want "hidden" are also from creation (i.e. the specific creater/degrader and/or flow file used?
```

if `save_pdfs` is set to True, the redshift posteriors will be saved for each estimation code with a filename given by the `estimation_results_base` keyword plus the name of the individual estimator.

The `estimators` keyword should contain a list of each of the RAIL estimation codes that you wish to run, and each of these should have a dictionary of options needed for each individual code named `run_params` (see documentation for each individual code for what specific run options are necessary, but note that the file `estimation_goldspike.yaml` has an example setup for BPZ_lite, FZBoost, and trainZ).
**NOTE**: we need documentation for the individual codes and yaml options needed to run each!

## Evaluation

`evaluation_goldspike.yaml` contains `evaluation_directory`, which specifies the location in which the evaluation outputs are stored. The summary of the metric statistics will be output to the filename specified in `results_file`.  `make_ks_figure`, if set to `True`, will generate a figure showing a visualization of the Kolmogorov-Smirnoff test using the PIT histogram and quantile-quantile plot for each of the codes from the `Estimation` stage, saved into the directory specified by `figure_directory`.  All other evaluations are run automatically as part of the goldenspike.py script.  If you wish to save the PIT values and point estimate redshifts into a parquet file for, e.g., combined figure making at a later stage, setting the `save_pit_mode_vals` flag to `True` will save a parquet file to the `evaluation_directory` with the code name and `_pit_mode_values.pq` suffix for each of the codes in estimation. 
