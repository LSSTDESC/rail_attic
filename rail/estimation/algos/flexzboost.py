"""
Example code that just spits out random numbers between 0 and 3
for z_mode, and Gaussian centered at z_mode with width
random_width*(1+zmode).
"""

import numpy as np
import h5py
import pandas as pd
import flexcode
from flexcode.regression_models import XGBoost
import pickle
from flexcode.loss_functions import cde_loss
from numpy import inf
from estimator import Estimator as BaseEstimation

def make_color_data(data_dict):
    """
    make a dataset consisting of the i-band mag and the five colors
    Returns:
    -------- 
    input_data: (nd-array)
      array of imag and 5 colors
    """
    input_data = data_dict['i_mag']
    bands = ['u','g','r','i','z','y']
    # make colors and append to input data
    for i in range(5):
        # replace the infinities with 28.0 just arbitrarily for now
        band1 = data_dict[f'{bands[i]}_mag']
        band2 = data_dict[f'{bands[i+1]}_mag']
        band1[band1 == inf] = 28.0
        band2[band2 == inf] = 28.0
        input_data = np.vstack((input_data, band1-band2))
        color_err = np.sqrt((data_dict[f'{bands[i]}_mag_err'])**2+ (data_dict[f'{bands[i+1]}_mag_err'])**2)
        input_data = np.vstack((input_data,color_err))
    return input_data.T


class FZBoost(BaseEstimation):
    """
    Subclass to implement a simple point estimate Neural Net photoz
    rather than actually predict PDF, for now just predict point zb
    and then put an error of width*(1+zb).  We'll do a "real" NN
    photo-z later.
    """
    def __init__(self,base_dict):
        """
        Parameters:
        -----------
        run_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """

        inputs = base_dict['run_params']
        
        self.zmin = inputs['zmin']
        self.zmax = inputs['zmax']
        self.nzbins = inputs['nzbins']
        self.trainfrac = inputs['trainfrac']
        self.bumpmin = inputs['bumpmin']
        self.bumpmax = inputs['bumpmax']
        self.nbump = inputs['nbump']
        self.sharpmin = inputs['sharpmin']
        self.sharpmax = inputs['sharpmax']
        self.nsharp = inputs['nsharp']
        self.max_basis = inputs['max_basis']
        self.basis_system = inputs['basis_system']
        self.regression_params = inputs['regression_params']
        super().__init__(base_dict)

    @staticmethod
    def partition_data(fz_data,sz_data,trainfrac):
        """
        make a random partition of the training data into training and 
        validation, validation data will be used to determine bump
        thresh and sharpen parameters.
        """
        nobs = fz_data.shape[0]
        ntrain = round(nobs*trainfrac)
        nvalidate = nobs - ntrain
        np.random.seed(1138) #set a specific seed for reproducibility
        perm = np.random.permutation(nobs)
        x_train = fz_data[perm[:ntrain],:]
        z_train = sz_data[perm[:ntrain]]
        x_val = fz_data[perm[ntrain:]]
        z_val = sz_data[perm[ntrain:]]
        return x_train,x_val,z_train,z_val


    def train(self):
        """
          train flexzboost model model
        """
        speczs = self.training_data['redshift_true']
        print("stacking some data...")
        color_data = make_color_data(self.training_data)
        train_data, val_data, train_sz, val_sz = self.partition_data(color_data,
                                                                speczs,
                                                                self.trainfrac)
        print("read in training data")
        model = flexcode.FlexCodeModel(XGBoost,max_basis=self.max_basis,
                                       basis_system=self.basis_system,
                                       z_min=self.zmin,z_max=self.zmax,
                                       regression_params=self.regression_params)
        print("fit the model...")
        model.fit(train_data,train_sz)
        bump_grid = np.linspace(self.bumpmin,self.bumpmax,self.nbump)
        print("finding best bump thresh...")
        bestloss = 9999
        for bumpt in bump_grid:
            model.bump_threshold=bumpt
            model.tune(val_data,val_sz)
            tmpcdes,z_grid = model.predict(val_data,n_grid=self.nzbins)
            tmploss = cde_loss(tmpcdes,z_grid,val_sz)
            if tmploss < bestloss:
                bestloss = tmploss
                bestbump = bumpt
        model.bump_threshold=bestbump
        print("finding best sharpen parameter...")
        sharpen_grid = np.linspace(self.sharpmin,self.sharpmax,self.nsharp)
        bestloss = 9999
        bestsharp = 9999
        for sharp in sharpen_grid:
            model.sharpen_alpha = sharp
            tmpcdes,z_grid = model.predict(val_data,n_grid=301)
            tmploss = cde_loss(tmpcdes,z_grid,val_sz)
            if tmploss < bestloss:
                bestloss = tmploss
                bestsharp = sharp
        model.sharpen_alpha=bestsharp
        self.model = model

        
    def run_photoz(self):
        print("running photoz's...")
        color_data = make_color_data(self.test_data)
        pdfs, z_grid = self.model.predict(color_data,n_grid=self.nzbins)
        
        self.zgrid = z_grid
        self.pz_pdf = pdfs

        self.zmode = [self.zgrid[np.argmax(pdf)] for pdf in self.pz_pdf]
