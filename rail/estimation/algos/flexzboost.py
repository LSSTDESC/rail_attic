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
from ceci.config import StageParameter as Param
from rail.estimation.estimator import Estimator, Trainer
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



class Train_FZBoost(Trainer):
    """ Train a FZBoost Estimator
    """
    name = 'Train_FZBoost'
    config_options = Trainer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          trainfrac=Param(float, 0.75,
                                          msg="fraction of training "
                                          "data to use for training (rest used for bump thresh "
                                          "and sharpening determination)"),
                          bumpmin=Param(float, 0.02,
                                        msg="minimum value in grid of "
                                        "thresholds checked to optimize removal of spurious "
                                        "small bumps"),
                          bumpmax=Param(float, 0.35,
                                        msg="max value in grid checked "
                                        "for removal of small bumps"),
                          nbump=Param(int, 20, msg="number of grid points in bumpthresh grid search"),
                          sharpmin=Param(float, 0.7, msg="min value in grid checked in optimal sharpening parameter fit"),
                          sharpmax=Param(float, 2.1, msg="max value in grid checked in optimal sharpening parameter fit"),
                          nsharp=Param(int, 15, msg="number of search points in sharpening fit"),
                          max_basis=Param(int, 35, msg="maximum number of basis funcitons to use in density estimate"),
                          basis_system=Param(str, 'cosine', msg="type of basis sytem to use with flexcode"),
                          bands=Param(str, 'ugrizy', msg="bands to use in estimation"),
                          regression_params=Param(dict, {'max_depth': 8, 'objective': 'reg:squarederror'},
                                                  msg="dictionary of options passed to flexcode, includes "
                                                  "max_depth (int), and objective, which should be set "
                                                  " to reg:squarederror"))


    def __init__(self, args, comm=None):
        """ Constructor
        Do Trainer specific initialization, then check on bands """
        Trainer.__init__(self, args, comm=comm)
        if not all(c in string.ascii_letters for c in self.config.bands):
            raise ValueError("'bands' option should be letters only (no spaces or commas etc)")

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

    def run(self):
        """Train flexzboost model model
        """
        training_data = self.get_data('input')[self.config.hdf5_groupname]
        speczs = training_data['redshift']
        print("stacking some data...")
        color_data = make_color_data(training_data, self.config.bands)
        train_dat, val_dat, train_sz, val_sz = self.split_data(color_data,
                                                               speczs,
                                                               self.config.trainfrac)
        print("read in training data")
        model = flexcode.FlexCodeModel(XGBoost, max_basis=self.config.max_basis,
                                       basis_system=self.config.basis_system,
                                       z_min=self.config.zmin, z_max=self.config.zmax,
                                       regression_params=self.config.regression_params)
        print("fit the model...")
        model.fit(train_dat, train_sz)
        bump_grid = np.linspace(self.config.bumpmin, self.config.bumpmax, self.config.nbump)
        print("finding best bump thresh...")
        bestloss = 9999
        for bumpt in bump_grid:
            model.bump_threshold = bumpt
            model.tune(val_dat, val_sz)
            tmpcdes, z_grid = model.predict(val_dat, n_grid=self.config.nzbins)
            tmploss = cde_loss(tmpcdes, z_grid, val_sz)
            if tmploss < bestloss:
                bestloss = tmploss
                bestbump = bumpt
        model.bump_threshold = bestbump
        print("finding best sharpen parameter...")
        sharpen_grid = np.linspace(self.config.sharpmin, self.config.sharpmax, self.config.nsharp)
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
        if self.config.save_train:
            with open(self.config.model_file, 'wb') as f:
                pickle.dump(file=f, obj=model, protocol=pickle.HIGHEST_PROTOCOL)


class FZBoost(Estimator):
    """FZBoost-based Estimator
    """
    name = 'FZBoost'
    config_options = Estimator.config_options.copy()
    config_options.update(nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          bands=Param(str, 'ugrizy', msg="bands to use in estimation"))

    def __init__(self, args, comm=None):
        """ Constructor:
        Do Estimator specific initialization """
        Estimator.__init__(self, args, comm=comm)
        if not all(c in string.ascii_letters for c in self.config.bands):
            raise ValueError("'bands' option should be letters only (no spaces or commas etc)")
        self.zgrid = None

    def run(self):
        test_data = self.get_data('input')[self.config.hdf5_groupname]
        color_data = make_color_data(test_data, self.config.bands)
        pdfs, z_grid = self.model.predict(color_data, n_grid=self.config.nzbins)
        self.zgrid = np.array(z_grid).flatten()
        qp_dstn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=pdfs))
        self.add_data('output', qp_dstn)
