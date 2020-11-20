"""
Example code that implements a simple Neural Net predictor
for z_mode, and Gaussian centered at z_mode with base_width
read in fromfile and pdf width set to base_width*(1+zmode).
"""

import numpy as np
# from numpy import inf
import sklearn.neural_network as sknn
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from rail.estimation.estimator import Estimator as BaseEstimation


def make_color_data(data_dict):
    """
    make a dataset consisting of the i-band mag and the five colors
    Returns:
    --------
    input_data: (nd-array)
    array of imag and 5 colors
    """
    input_data = data_dict['mag_i_lsst']
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    # make colors and append to input data
    for i in range(5):
        # replace the non-detect 99s with 28.0 just arbitrarily for now
        band1 = data_dict[f'mag_{bands[i]}_lsst']
        # band1err = data_dict[f'mag_err_{bands[i]}_lsst']
        band2 = data_dict[f'mag_{bands[i+1]}_lsst']
        # band2err = data_dict[f'mag_err_{bands[i+1]}_lsst']
        # for j,xx in enumerate(band1):
        #    if np.isclose(xx,99.,atol=.01):
        #        band1[j] = band1err[j]
        #        band1err[j] = 1.0
        # for j,xx in enumerate(band2):
        #    if np.isclose(xx,99.,atol=0.01):
        #        band2[j] = band2err[j]
        #        band2err[j] = 1.0
        input_data = np.vstack((input_data, band1-band2))
    return input_data.T


def regularize_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    regularized_data = scaler.transform(data)
    return regularized_data


class simpleNN(BaseEstimation):
    """
    Subclass to implement a simple point estimate Neural Net photoz
    rather than actually predict PDF, for now just predict point zb
    and then put an error of width*(1+zb).  We'll do a "real" NN
    photo-z later.
    """
    def __init__(self, base_config, config_dict):
        """
        Parameters:
        -----------
        run_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """

        super().__init__(base_config=base_config, config_dict=config_dict)
        inputs = self.config_dict['run_params']

        self.width = inputs['width']
        self.zmin = inputs['zmin']
        self.zmax = inputs['zmax']
        self.nzbins = inputs['nzbins']
        np.random.seed(71)

    def inform(self):
        """
          train the NN model
        """
        speczs = self.training_data['redshift']
        print("stacking some data...")
        color_data = make_color_data(self.training_data)
        input_data = regularize_data(color_data)
        simplenn = sknn.MLPRegressor(hidden_layer_sizes=(12, 12),
                                     activation='tanh', solver='lbfgs')
        simplenn.fit(input_data, speczs)
        self.model = simplenn

    def estimate(self, test_data):
        color_data = make_color_data(test_data)
        input_data = regularize_data(color_data)
        zmode = np.round(self.model.predict(input_data), 3)
        pdfs = []
        widths = self.width * (1.0+zmode)
        self.zgrid = np.linspace(self.zmin, self.zmax, self.nzbins)
        for i, zb in enumerate(zmode):
            pdfs.append(norm.pdf(self.zgrid, zb, widths[i]))
        pz_dict = {'zmode': zmode, 'pz_pdf': pdfs}
        return pz_dict
