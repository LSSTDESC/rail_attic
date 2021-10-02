# Golden Spike Example Script.

"Golden spike" is meant to demonstrate a full "end-to-end" running of the RAIL infrastructure, something you might do to compare the performance of photo-z codes a la the PhotoZDC1 challenge described in [Schmidt et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.1587S/abstract).  There are three main modules that will be exercised within golden spike: mock data will be created with the Creation module, codes will be run with the Estimation module, and metrics computed with the Evaluation module.

You must have RAIL and all dependencies installed locally in order to run goldenspike.  In addition, if you do not have either a pre-trained pzflow object or a local set of data with which to train the RAIL/creation creator that will create your mock data, then you need to have DESC's GCRCatalogs installed so that the script can fetch data with which to train the flow.  GCRCatalogs is installed on cori at NERSC, so in this latter case it is recommended that you run the script on cori.

The main script is `goldenspike.py` which reads is controlled by the many options in three yaml files, one for each major stage: `creation_goldspike.yaml`, `estimation_goldspike.yaml`, and `evaluation_goldspike.yaml`.  Below we will go through some of the features of each stage:
NOTE: could switch to grab these on command line rather than hardcoded names, ask others for preference.


## Creation
The major functionality of the creation module is to create mock data with which to test the set of estimators, which also produces "true" redshift posteriors for each mock galaxy.  There are three options for generating mock data: 1) starting with a pre-trained flow; 2) loading photometry+redshift data from file; 3) querying a healpix pixel from GCRCatalogs (if installed, so usually when running at NERSC).  

## Data used as input to RAIL Creator
In `creation_goldspike.yaml` setting `has_flow: True` will read in the pre-trained flow from the pickle file specified with `flow_file`.  

If you instead wish to read in data from file, setting `use_local_dat: True will read in the photometry and redshift info from the file specified in `local_flow_data_file` (there is an example data file named `data/test_flow_data.pq` that will work as a default.  Note that you must set `flow_columns` to an array that includes only the columns that you wish to input into the flow, as extra columns may lead to lower quality fits, as the neural flow will spend time and coefficients trying to emulate the larger space that includes the extra columns.

If `has_flow` and `use_local_data` are both False, the script will instead use GCRCatalogs to grab one healpix pixel specified by `hpix` and keep a fraction `frac` of that data, where frac is an integer, e.g. frac=10 will keep 1/10th of the data while frac=50 will keep only 1/50th of the data on which to train the flow.

Once the flow is created, you can choose whether to apply a degrader with the `use_degrader` option, if it is set to False then the same creator is used for both test and training data 
**TODO**: switch to generic degrader and options dictionary
The script then creates training data (using the undegraded creator) and test data (using the degraded creator, if applied), you set the number of each generated with `N_test_gals` and `N_train_gals` in the yaml file.
The "true posteriors" for each galaxy are evaluated on a grid, you set this grid with `zmin`, `zmax`, and `nzbins` in the yaml file, and set the name used for the redshift column using `z_column`.

If you wish to save the true posteriors to file, set `save_ensemble: True` and they will be written as a qp Ensemble to the filename specified in `ensemble_file`.  Similarly, if you wish to save the flow that you trained, `save_flow: True` will save the flow to the filename specified in `saved_flow_file`

The test and train datasets that were sampled from the flow and will be input to the RAIL estimators are written to the files specified by the values set in `test_filename` and `train_filename`

