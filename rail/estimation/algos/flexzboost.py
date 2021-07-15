"""
Implementation of the FlexZBoost algorithm, uses training data and
XGBoost to learn the relation, split training data into train and
validation set and find best "bump_thresh" (eliminate small peaks in
p(z) below threshold) and sharpening parameter (determines peakiness of
p(z) shape) via cde-loss over a grid.
"""

import numpy as np
import pickle
import flexcode
import qp
from flexcode.regression_models import XGBoost
from flexcode.loss_functions import cde_loss
# from numpy import inf
from rail.estimation.estimator import Estimator as BaseEstimation
from rail.estimation.utils import check_and_print_params
import string

def make_color_data(data_dict, bands):
    """
    make a dataset consisting of the i-band mag and the five colors.

    Parameters
    -----------
    data_dict : `ndarray`
      array of magnitudes and errors, with names mag_{bands[i]}_lsst 
      and mag_err_{bands[i]}_lsst respectively.

    Returns
    --------
    input_data : `ndarray`
      array of imag and 5 colors
    """
    input_data = data_dict['mag_i_lsst']
    # make colors and append to input data
    for i in range(len(bands)-1):
        band1 = data_dict[f'mag_{bands[i]}_lsst']
        band1err = data_dict[f'mag_err_{bands[i]}_lsst']
        band2 = data_dict[f'mag_{bands[i+1]}_lsst']
        band2err = data_dict[f'mag_err_{bands[i+1]}_lsst']
        for j, xx in enumerate(band1):
            if np.isclose(xx, 99., atol=.01):
                band1[j] = band1err[j]
                band1err[j] = 1.0
        for j, xx in enumerate(band2):
            if np.isclose(xx, 99., atol=0.01):  #pragma: no cover
                band2[j] = band2err[j]
                band2err[j] = 1.0

        input_data = np.vstack((input_data, band1-band2))
        color_err = np.sqrt((band1err)**2 + (band2err)**2)
        input_data = np.vstack((input_data, color_err))
    return input_data.T


def_param = {'run_params': {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                            'trainfrac': 0.75, 'bumpmin': 0.02,
                            'bumpmax': 0.35, 'nbump': 20,
                            'sharpmin': 0.7, 'sharpmax': 2.1,
                            'nsharp': 15, 'max_basis': 35,
                            'basis_system': 'cosine',
                            'bands': 'ugrizy',
                            'regression_params': {'max_depth': 8,
                                                  'objective':
                                                  'reg:squarederror'
                                                  },
                            'inform_options': {'save_train': True,
                                               'load_model': False,
                                               'modelfile':
                                               "FZBoost.pkl"
                                               }
                            }
             }


desc_dict = {'zmin': "zmin (float): min value for z grid",
             'zmax': "zmax (float): max value for z grid",
             'nzbins': "nzbins (int): number of z bins",
             'bands': "bands (str): bands to use in estimation",
             'trainfrac': "trainfrac (float): fraction of training "
             "data to use for training (rest used for bump thresh "
             "and sharpening determination)",
             'bumpmin': "bumpmin (float): minimum value in grid of "
             "thresholds checked to optimize  removal of spurious "
             "small bumps",
             'bumpmax': "bumpmax (float) max value in grid checked "
             "for removal of small bumps",
             'nbump': "nbump (int): number of grid points in "
             "bumpthresh grid search",
             'sharpmin': "sharpmin (float): min value in grid "
             "checked for optimal sharpening parameter fit",
             'sharpmax': "sharpmax (float): max value in grid "
             "checked in optimal sharpening parameter fit",
             'nsharp': "nsharp (int): number of search points in "
             "sharpening fit",
             'max_basis': "max_basis (int): maximum number of "
             "basis funcitons to use in density estimate",
             'basis_system': "basis_system (str): type of "
             "basis sytem to use with flexcode",
             'regression_params': "regression_params (dict): "
             "dictionary or options passed to flexcode, includes "
             "max_depth (int), and objective, which should be set "
             " to reg:squarederror",
             'inform_options': "inform_options (dict): a dictionary "
             "of options for loading and storing of a pretrained "
             "model.  This includes:\n modelfile (str): the filename"
             " to save or load\n save_train (bool): boolean to set "
             "whether to save a trained model\n load_model (bool): "
             "boolean to set whether to load a pretrained model"
             }


class FZBoost(BaseEstimation):
    """
    Subclass to implement a simple point estimate Neural Net photoz
    rather than actually predict PDF, for now just predict point zb
    and then put an error of width*(1+zb).  We'll do a "real" NN
    photo-z later.
    """
    def __init__(self, base_dict, config_dict='None'):
        """
        Parameters
        -----------
        base_dict: dict
          dictionary of variables from base.yaml-type file
        config_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """
        if config_dict == "None":
            print("No config file supplied, using default parameters")
            config_dict = def_param
        config_dict = check_and_print_params(config_dict, def_param,
                                             desc_dict)
        super().__init__(base_dict, config_dict)

        inputs = config_dict['run_params']

        self.zmin = inputs['zmin']
        self.zmax = inputs['zmax']
        self.nzbins = inputs['nzbins']
        self.trainfrac = inputs['trainfrac']
        self.bumpmin = inputs['bumpmin']
        self.bumpmax = inputs['bumpmax']
        self.nbump = inputs['nbump']
        self.bands = inputs['bands']
        self.sharpmin = inputs['sharpmin']
        self.sharpmax = inputs['sharpmax']
        self.nsharp = inputs['nsharp']
        self.max_basis = inputs['max_basis']
        self.basis_system = inputs['basis_system']
        self.regress_params = inputs['regression_params']
        self.inform_options = inputs['inform_options']

        if not all(c in string.ascii_letters for c in self.bands):
            raise ValueError("'bands' option should be letters only (no spaces or commas etc)")

        if 'save_train' in inputs['inform_options']:
            try:
                self.modelfile = self.inform_options['modelfile']
                print(f"using modelfile {self.modelfile}")
            except KeyError:  #pragma: no cover
                defModel = "default_model.out"
                print(f"name for model not found, will save to {defModel}")
                self.inform_options['modelfile'] = defModel

    @staticmethod
    def split_data(fz_data, sz_data, trainfrac):
        """
        make a random partition of the training data into training and
        validation, validation data will be used to determine bump
        thresh and sharpen parameters.
        """
        nobs = fz_data.shape[0]
        ntrain = round(nobs * trainfrac)
        # set a specific seed for reproducibility
        np.random.seed(1138)
        perm = np.random.permutation(nobs)
        x_train = fz_data[perm[:ntrain], :]
        z_train = sz_data[perm[:ntrain]]
        x_val = fz_data[perm[ntrain:]]
        z_val = sz_data[perm[ntrain:]]
        return x_train, x_val, z_train, z_val

    def inform(self, training_data):
        """
          train flexzboost model model
        """
        speczs = training_data['redshift']
        print("stacking some data...")
        color_data = make_color_data(training_data, self.bands)
        train_dat, val_dat, train_sz, val_sz = self.split_data(color_data,
                                                               speczs,
                                                               self.trainfrac)
        print("read in training data")
        model = flexcode.FlexCodeModel(XGBoost, max_basis=self.max_basis,
                                       basis_system=self.basis_system,
                                       z_min=self.zmin, z_max=self.zmax,
                                       regression_params=self.regress_params)
        print("fit the model...")
        model.fit(train_dat, train_sz)
        bump_grid = np.linspace(self.bumpmin, self.bumpmax, self.nbump)
        print("finding best bump thresh...")
        bestloss = 9999
        for bumpt in bump_grid:
            model.bump_threshold = bumpt
            model.tune(val_dat, val_sz)
            tmpcdes, z_grid = model.predict(val_dat, n_grid=self.nzbins)
            tmploss = cde_loss(tmpcdes, z_grid, val_sz)
            if tmploss < bestloss:
                bestloss = tmploss
                bestbump = bumpt
        model.bump_threshold = bestbump
        print("finding best sharpen parameter...")
        sharpen_grid = np.linspace(self.sharpmin, self.sharpmax, self.nsharp)
        bestloss = 9999
        bestsharp = 9999
        for sharp in sharpen_grid:
            model.sharpen_alpha = sharp
            tmpcdes, z_grid = model.predict(val_dat, n_grid=301)
            tmploss = cde_loss(tmpcdes, z_grid, val_sz)
            if tmploss < bestloss:
                bestloss = tmploss
                bestsharp = sharp
        model.sharpen_alpha = bestsharp
        self.model = model
        if self.inform_options['save_train']:
            with open(self.inform_options['modelfile'], 'wb') as f:
                pickle.dump(file=f, obj=model,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def estimate(self, test_data):
        color_data = make_color_data(test_data, self.bands)
        pdfs, z_grid = self.model.predict(color_data, n_grid=self.nzbins)
        self.zgrid = np.array(z_grid).flatten()
        if self.output_format == 'qp':
            qp_dstn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid,
                                                       yvals=pdfs))
            return qp_dstn
        else:
            zmode = np.array([self.zgrid[np.argmax(pdf)]
                              for pdf in pdfs]).flatten()
            pz_dict = {'zmode': zmode, 'pz_pdf': pdfs}
            return pz_dict
