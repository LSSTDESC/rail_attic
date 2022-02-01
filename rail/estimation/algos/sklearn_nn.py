"""
Example code that implements a simple Neural Net predictor
for z_mode, and Gaussian centered at z_mode with base_width
read in fromfile and pdf width set to base_width*(1+zmode).
"""

import numpy as np
import pickle
# from numpy import inf
import sklearn.neural_network as sknn
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from ceci.config import StageParameter as Param
from rail.estimation.estimator import Estimator, Trainer
import string
import qp


def make_color_data(data_dict, bands):
    """
    make a dataset consisting of the i-band mag and the five colors

    Returns
    --------
    input_data: `ndarray` array of imag and 5 colors
    """
    input_data = data_dict['mag_i_lsst']
    # make colors and append to input data
    for i in range(len(bands)-1):
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


class Train_SimpleNN(Trainer):
    """
    Subclass to implement a simple point estimate Neural Net photoz
    rather than actually predict PDF, for now just predict point zb
    and then put an error of width*(1+zb).  We'll do a "real" NN
    photo-z later.
    """

    name = 'Train_SimpleNN'
    config_options = Trainer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          width=Param(float, 0.05, msg="The ad hoc base width of the PDFs"),
                          bands=Param(str, 'ugrizy', msg="bands to use in estimation"),
                          max_iter=Param(int, 500,
                                         msg="max number of iterations while "
                                         "training the neural net.  Too low a value will cause an "
                                         "error to be printed (though the code will still work, just"
                                          "not optimally)"))


    def __init__(self, args, comm=None):
        Trainer.__init__(self, args, comm=comm)
        if not all(c in string.ascii_letters for c in self.config.bands):
            raise ValueError("'bands' option should be letters only (no spaces or commas etc)")
        np.random.seed(71)

    def run(self):
        """
          train the NN model
        """
        training_data = self.get_data('input')                       
        speczs = training_data['redshift']
        print("stacking some data...")
        color_data = make_color_data(training_data, self.config.bands)
        input_data = regularize_data(color_data)
        simplenn = sknn.MLPRegressor(hidden_layer_sizes=(12, 12),
                                     activation='tanh', solver='lbfgs',
                                     max_iter=self.config.max_iter)
        simplenn.fit(input_data, speczs)
        model = simplenn
        if self.config.save_train:
            with open(self.config.model_file, 'wb') as f:
                pickle.dump(file=f, obj=model,
                            protocol=pickle.HIGHEST_PROTOCOL)


class SimpleNN(Estimator):
    """
    Subclass to implement a simple point estimate Neural Net photoz
    rather than actually predict PDF, for now just predict point zb
    and then put an error of width*(1+zb).  We'll do a "real" NN
    photo-z later.
    """
    name = 'SimpleNN'
    config_options = Estimator.config_options.copy()
    config_options.update(width=Param(float, 0.05, msg="The ad hoc base width of the PDFs"),
                          bands=Param(str, 'ugrizy', msg="bands to use in estimation"))
    
    def __init__(self, args, comm=None):
        Estimator.__init__(self, args, comm=comm)
                
    def run(self):
        test_data = self.get_data('input')
        color_data = make_color_data(test_data, self.config.bands)
        input_data = regularize_data(color_data)
        zmode = np.round(self.model.predict(input_data), 3)
        pdfs = []
        widths = self.config.width * (1.0+zmode)

        qp_dstn = qp.Ensemble(qp.stats.norm, data=dict(loc=zmode, scale=widths))
        self.add_data('output', qp_dstn)

