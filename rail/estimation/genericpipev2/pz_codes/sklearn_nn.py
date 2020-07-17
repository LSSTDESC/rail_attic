"""
Example code that just spits out random numbers between 0 and 3
for z_mode, and Gaussian centered at z_mode with width
random_width*(1+zmode).
"""

import numpy as np
from numpy import inf
import sklearn.neural_network as sknn
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from .tomo import Tomographer
from .base import BaseEstimation

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
    return input_data.T

def regularize_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    regularized_data  = scaler.transform(data)
    return regularized_data



class simpleNN(Tomographer,BaseEstimation):
#class randomPZ(Tomographer): 
   
    def __init__(self,inputs):
        """
        Parameters:
        -----------
        run_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """
        self.width = inputs['width']
        self.zmin = inputs['zmin']
        self.zmax = inputs['zmax']
        self.nzbins = inputs['nzbins']
        super().__init__(inputs)


    #def regularize_data(data):
    #    scaler = StandardScaler()
    #    scaler.fit(data)
    #    regularized_data  = scaler.transform(data)
    #    return regularized_data

    #def make_color_data(data_dict):
    #    """
    #      make a dataset consisting of the i-band mag and the five colors
    #    Returns:
    #    --------
    #      input_data: (nd-array)
    #      array of imag and 5 colors
    #    """
    #    input_data = data_dict['i_mag']
    #    bands = ['u','g','r','i','z','y']
    #    # make colors and append to input data 
    #    for i in range(5):
    #        input_data = np.vstack((input_data, data_dict[f'{bands[i]}_mag'] - data_dict[f'{bands[i+1]}_mag']))
    #    return input_data

    def train(self):
        """
          train the NN model
        """
        speczs = self.training_data['redshift_true']
        print("stacking some data...")
        color_data = make_color_data(self.training_data)
        input_data = regularize_data(color_data)
        simplenn = sknn.MLPRegressor(hidden_layer_sizes=(12,12),activation='tanh',solver='lbfgs')
        simplenn.fit(input_data,speczs)
        self.model = simplenn
        


    def run_photoz(self):
        print("running photoz's...")
        color_data = make_color_data(self.test_data)
        input_data = regularize_data(color_data)
        self.zmode = self.model.predict(input_data)
        pdf = []
        widths = self.width*(1.0+self.zmode)
        self.zgrid = np.linspace(self.zmin,self.zmax,self.nzbins)
        for i,zb in enumerate(self.zmode):
            pdf.append(norm.pdf(self.zgrid,zb,widths[i]))
        self.pz_pdf = pdf
