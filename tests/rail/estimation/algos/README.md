## Available Estimators

There are multiple estimators available for use with RAIL, this README gives a brief description of each and documents the configuration parameters necessary to run each algorithm as defined in config dictionary that is passed in when the algorithm is instantiated.  Example configuration files can be found in `rail/examples/estimation/configs` for each algorithm.  One way to quickly get started on running an estimator is to use the `main.py` script located in `rail/examples/estimation` with `python main.py [configs/specific_code_config.yaml] [specific_base_params.yaml]`.  See the README in `RAIL/examples/estimation` for more details.  Or, see the `RAIL/examples/estimation/RAIL_estimation_demo.ipynb` notebook.

# Common configuration parameters
Nearly all of the codes require some basic config parameters to define how the redshifts are stored, how a trained model/prior is named/loaded/saved, etc..., we will list a few of these parameters below:
- `zmin`: the mininum redshift for which to estimate a photo-z.
- `zmax`: the maximum redshift for which to estimate a photo-z.
- `nzbins`: the number of bins on which to estimate a photo-z.
The usual procedure for a gridded parameterization is to define the redshift evaluation grid as `numpy.linspace(zmin, zmax, nzbins)`.

- `inform_options`: This is a dictionary consisting of three entries that control the behavior of the `inform` option for a code, i.e. saving a trained model/prior after inform has been run, or loading a pre-existing model so that the inform stage can be skipped.  The entries are:
- `modelfile`: a string consisting of the filename for the loaded/saved model file.
- `load_model`: boolean, if True codes should skip inform and load a pretrained model from the filename specified in `modelfile`.
- `save_train`: boolean, if True codes should save the model as computed during the running of `inform` to the filename in `modelfile`.


# BPZ_Lite

This is a python3 version of the popular BPZ (Benitez 2000) template-based estimator that employs a Bayesian prior P(z|m, T) where m is apparent magnitude and T is type/SED.  Some documentation for BPZ is available on Dan Coe's BPZ website at: https://www.stsci.edu/~dcoe/BPZ/ 

The specific parameters needed in order to run BPZ_lite:
- `dz`: the delta z to define the redshift grid, which is defined as np.arange(zmin, zmax+dz, dz) rather than via numpy.linspace.  This is to be consistent with the original BPZ code.

- `columns_file`: string containing the path the columns file, which consists of a columnar list of the FILTER names, name of the magnitude and error columns in the input data, whether the magnitudes are in AB or Vega magnitudes, an error that will be added in quadrature to the magnitude errors, and a potential zero point offset for the magnitudes.  For more details see the BPZ documentation.

- `spectra_file`: string containing the path to the ".list" file that contains a list of the SED templates to be used by BPZ.

- `madau_flag`: string containing `yes` or `no` that controls whether a model for Madau (1995) reddenning associated with IGM absorption will be applied when computing synthetic fluxes.

- `bands`: a string containing the single-letter filters as used for LSST, i.e. "ugrizy".  This parameter is used by TXPipe when selecting subsets of bands for which photometry will be used.

- `prior_band`: single character, e.g. "i" specifying which band the prior is specified in terms of.

- `prior_file`: string consisting of the name of the prior file, which should be located in the BPZ directory.  The code will tack on a preamble and file extension, so while the actual file will be named, e.g "prior_hdfn_gen.py", the actual entry should be "hdfn_gen", omitting the "prior_" prefix and ".py extension.

- `p_min`: float that specifies a minimum posterior probability.  Any values in the PDF that fall below this value will be set to 0.0 to reduce storage.

- `gauss_kernel`: float that specifies the width of a Gaussian kernel that will smooth the resulting PDF to reduce small scale noise.

- `zp_errors`: an array with a length that matches the number of FILTERS in the .columns file that specifies an error to be added in quadrature to each magnitude error.

- `mag_err_min`: float that sets a minimum value for each magnitude error, if a magnitude error is below this value it will be set to this value.


# FlexZBoost
FlexZBoost is a machine learning-based code that arrives at a PDF estimate via conditional density estimation.  The code takes in training data, reserves a portion for validation, and uses that validation set to find best-fit values for a "bump_threshold" (trimming residual small scale peaks in the PDF), and "sharpening" (convolution that can narrow/widen the PDF) parameters.  Best fit values for bump_thresh and sharpening are found by minimizing the CDE Loss function.  Internally FlexZBoost uses a set of cosine or Fourier basis functions; however, the current code is implemented such that it outputs on a redshift grid defined by zmin, zmax, and nzbins.

parameters needed to run FlexZBoost:
- `trainfrac`: float specifying what fraction of the data to be used in training.  The remainder is used for validation in finding bump_thresh and sharpen params.

- `bumpmin`: float, minimum value for which to check bump removal.

- `bumpmax`: float, max value for which to check for bump removal.

- `nbump`: int, number of values to check in bump_thresh grid.

- `sharpmin`, `sharpmax, `nsharp`: similar to bumpmin, max, and nbump, defines the grid of sharpening parameters over which to search.

- `max_basis`: int, the maximum number of basis functions to use in determining the PDF.

- `basis_system`: string listing the name of the basis set used, see FlexZBoost documentation for more details.

- `bands`: string, list of the single-letter band names used, e.g. "ugrizy", this is again employed by TXPipe for band selections.

- `regression_params`: dictionary with two options needed by FlexZBoost to control the regression, the two options are:
- `max_depth`: integer, depth of the regression

- `objective`: string that names the objective function used within xgboost.  This is needed to replace a change in the name in updated versions of xgboost, and should be set to "reg:squarederror".

# KNearNeighPDF
KNearNeighPDF is a very fast, very simple K nearest neighbor estimator.  It ignores photometric errors for the sake of speed, and uses only a single magnitude band and all colors to build a KD-tree, then finds K nearest neighbors and constructs a PDF based on those neighbors with Gaussians where the height of each Gaussian is set by the Euclidean distance in magnitude/color space, and the width of the Gaussian is a free parameter *sigma*.  Best fit values for sigma and K (the number of neighbors) can be optimized by searching a grid of these two parameters and minimizing the CDELoss for a portion of the training data reserved via the `trainfrac` value in the config file.  
In the config file the input training data is split into training and validation, keeping `trainfrac` to construct a KD-tree, and the remainder used as a validation set to compute the CDE loss.  The parameter values that are optimized over are set for sigma via: `sigma_grid_min`, `sigma_grid_max`, `ngrid_sigma`, (i.e. np.linspace(sigma_grid_min, sigma_grid_max, ngrid_sigma); and, for how many Neighbors, K, `nneigh_min`, and `nneigh_max` (i.e. range(nneigh_min, nneigh_max+1).  The combination of K and sigma with the minimal loss are stored and used for the final computation, and the final KD-tree is remade using the full training data set.
The  parameter options for KNearNeighPDF are:

-`trainfrac`: float, the fraction of training data used to construct the KD-tree, the remainder used to set best sigma and best K in the loss optimization described above.

-`random_seed`: int, integer to set the numpy random seed for reproducibility

-`column_names`: array if strings, the column names to be used in NN as named in the input dictionary of data.

-`ref_column_name`: string, name of the magnitude column to be used by the flow (other magnitude columms will be converted to colors).

-`redshift_column_name`: str, name of redshift column

- `mag_limits`: dict, dictionary with the magnitude column names, where each entry contains a float specifying the 1 sigma magnitude limit in that band.  "Non-detections" will be replaced with the 1 sigma magnitude limit, and a magnitude error of 0.75257.

-`sigma_grid_min`: float, minimum value of sigma for grid check

-`sigma_grid_max`: float, maximum value of sigma for grid check

-`leaf_size`: int, min leaf size for KDTree

-`nneigh_min`: int, min number of near neighbors to use for PDF fit

-`nneigh_max`: int, max number of near neighbors to use ofr PDF fit


# PZFlowPDF
PZFlowPDF implements an estimator using the [pzflow](https://github.com/jfcrenshaw/pzflow) package (which is extensively used in RAIL/creation).  The training data is used to train a normalizing flow, which can then provide a posterior estimate of redshift.  The base flow trains the flow using only one magnitude and adjacent colors, however there is an option, `include_mag_errors` that will marginalize over the supplied magnitude error by drawing N samples from the magnitude error distribution (assumed Gaussian for now) where N is given by the `n_error_samples` config option.  Note that marginalizing over the magnitude errors via sampling does come with an increase in computational cost.  The full list of parameter options for PZFlowPDF are:

- `flow_seed`: int, seed to feed into the flow to set random number generation

- `ref_column_name`: string, name of the magnitude column to be used by the flow (other magnitude columms will be converted to colors)

- `column_names`: names for the other magnitude columns, in addition to `ref_column_name`

- `mag_limits`: dict, dictionary with the magnitude column names, where each entry contains a float specifying the 1 sigma magnitude limit in that band.  "Non-detections" will be replaced with the 1 sigma magnitude limit, and a magnitude error of 0.75257.

- `include_mag_errors`: Bool, sets whether to sample N samples (given by `n_error_samples`) from the magnitude error distribution in order to margninalize over the effect of the magnitude errors

- `error_names_dict`: dictionary containing the names of the magnitude columns as keys and names of corresponding magnitude error columns as values.  This is needed to keep the magnitude error columns separate from the magntidues, but still tracked properly internal to the code.

- `n_error_samples`: integer, the number of samples to draw from each magnitude error distribution when marginalizing

- `soft_sharpness`: integer, sets softness for flow

- `soft_idx_col`: integer, column index

- `redshift_column_name`: string, the name of the redshift column in the input file

- `num_training_epochs`: int, the number of iteration steps during the loss minimization when training the flow

# randomPZ
randomPZ is not a real photo-z code, it is a placeholder demo code used to demonstrate the overall structure that an estimator subclass should have. It assigns a random redshift and outputs a simple Gaussian at random for each PDF.  But, for completion the parameters necessary are:
- `rand_with`: bin width of redshift grid.

- `rand_zmin`: minimum redshift for grid

- `rand_zmax`: maximum redshift for grid

# simpleNN
Another "demo" photo-z algorithm, this subclass uses sklearn's neural_network to create a simple point estimate for redshift, and outputs a Gaussian redshift estimate based soley on an ad-hoc `width` parameter specified by the user.  It is *not* a fully functional code, and should again be though of more for demonstration.  In the future we will implement a more sophisticated NN-based photo-z and likely remove this demo.

- `width`: width of the PDFs, where the output Gaussian will be assigned a width of width*(1+zpoint).

- `max_iter`: int, maximum number of iterations in the neural net training.

- `bands`: string of the single-letter filter names, e.g. "ugrizy", again used by TXPipe.

# trainZ
trainZ is our "pathological" photo-z estimator, it calculates the N(z) histogram of the training data, normalizes this, and outputs this as a redshift estimate for each galaxy in the test sample.  No parameters beyond the zmin, zmax, nzbins, and inform options are necessary to run the code.  As every PDF will be identical, running this estimator for a large number of objects can be a waste of space, and you might want to consider just storing the normalized N(z) separately.
