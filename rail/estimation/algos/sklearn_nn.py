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
from rail.estimation.estimator import Estimator as BaseEstimation
from rail.estimation.utils import check_and_print_params
import string
import qp


def_param = {'run_params': {'width': 0.05, 'zmin': 0.0,
                            'zmax': 3.0, 'nzbins': 301,
                            'max_iter': 500,
                            'bands': 'ugrizy',
                            'inform_options': {'save_train': False,
                                               'load_model': False,
                                               'modelfile':
                                               'simpleNN.pkl'
                                               }}}


descrip_dict = {'width': "width (float): The ad hoc base width of the PDFs",
                'zmin': "zmin (float): The minimum redshift of the z grid",
                'zmax': "zmax (float): The maximum redshift of the z grid",
                'nzbins': "nzbins (int): The number of points in the z grid",
                'bands': "bands (str): bands to use in estimation",
                'max_iter': "max_iter (int): max number of iterations while "
                "training the neural net.  Too low a value will cause an "
                "error to be printed (though the code will still work, just"
                "not optimally)",
                'inform_options': "inform_options (dict): a dictionary of "
                "options for loading and storing of the pretrained model.  "
                "This includes:\n modelfile (str): the filename to save or "
                "load a trained model to/from\n save_train (bool): boolean "
                "to set whether to save a trained model\n load_model (bool): "
                "boolean to set whether to load a pretrained model"
                }


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


class simpleNN(BaseEstimation):
    """
    Subclass to implement a simple point estimate Neural Net photoz
    rather than actually predict PDF, for now just predict point zb
    and then put an error of width*(1+zb).  We'll do a "real" NN
    photo-z later.
    """
    def __init__(self, base_config, config_dict='None'):
        """
        Parameters
        -----------
        run_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """
        if config_dict == "None":
            print("No config file supplied, using default parameters")
            config_dict = def_param
        config_dict = check_and_print_params(config_dict, def_param,
                                             descrip_dict)

        super().__init__(base_config=base_config, config_dict=config_dict)
        inputs = self.config_dict['run_params']

        self.width = inputs['width']
        self.zmin = inputs['zmin']
        self.zmax = inputs['zmax']
        self.bands = inputs['bands']
        self.nzbins = inputs['nzbins']
        self.maxiter = inputs['max_iter']
        self.inform_options = inputs['inform_options']
        if not all(c in string.ascii_letters for c in self.bands):
            raise ValueError("'bands' option should be letters only (no spaces or commas etc)")
        if 'save_train' in inputs['inform_options']:
            try:
                self.modelfile = self.inform_options['modelfile']
            except KeyError:  #pragma: no cover
                defModel = "default_model.out"
                print(f"name for model not found, will save to {defModel}")
                self.inform_options['modelfile'] = defModel

        np.random.seed(71)

    def inform(self, training_data):
        """
          train the NN model
        """
        speczs = training_data['redshift']
        print("stacking some data...")
        color_data = make_color_data(training_data, self.bands)
        input_data = regularize_data(color_data)
        simplenn = sknn.MLPRegressor(hidden_layer_sizes=(12, 12),
                                     activation='tanh', solver='lbfgs',
                                     max_iter=self.maxiter)
        simplenn.fit(input_data, speczs)
        self.model = simplenn
        if self.inform_options['save_train']:
            with open(self.inform_options['modelfile'], 'wb') as f:
                pickle.dump(file=f, obj=self.model,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def estimate(self, test_data):
        color_data = make_color_data(test_data, self.bands)
        input_data = regularize_data(color_data)
        zmode = np.round(self.model.predict(input_data), 3)
        pdfs = []
        widths = self.width * (1.0+zmode)

        if self.output_format == 'qp':
            qp_dstn = qp.Ensemble(qp.stats.norm, data=dict(loc=zmode,
                                                           scale=widths))
            return qp_dstn
        else:
            self.zgrid = np.linspace(self.zmin, self.zmax, self.nzbins)
            for i, zb in enumerate(zmode):
                pdfs.append(norm.pdf(self.zgrid, zb, widths[i]))
            pz_dict = {'zmode': zmode, 'pz_pdf': pdfs}
            return pz_dict
