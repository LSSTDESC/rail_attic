"""
Example code that just spits out random numbers between 0 and 3
for z_mode, and Gaussian centered at z_mode with width
random_width*(1+zmode).
"""

import numpy as np
from scipy.stats import norm
from rail.estimation.estimator import Estimator as BaseEstimation
from rail.estimation.utils import check_and_print_params
import qp
import pprint
import json

def_param = {'run_params': {'rand_width': 0.025, 'rand_zmin': 0.0,
                            'rand_zmax': 3.0, 'nzbins': 301,
                            'inform_options': {'save_train': False,
                                               'load_model': False,
                                               'modelfile':
                                               'randommodel.pkl'
                                               }}}


desc_dict = {'rand_width': "rand_width (float): ad hock width of PDF",
             'rand_zmin': "rand_zmin (float): min value for z grid",
             'rand_zmax': "rand_zmax (float): max value for z grid",
             'nzbins': "nzbins (int): number of z bins",
             'inform_options': "inform_options: (dict): a "
             "dictionary of options for loading and storing of "
             "the pretrained model.  This includes:\n "
             "modelfile:(str) the filename to save or load a "
             "trained model from.\n save_train:(bool) boolean to "
             "set whether to save a trained model.\n "
             "load_model:(bool): boolean to set whether to "
             "load a trained model from filename modelfile"
             }


class randomPZ(BaseEstimation):

    def __init__(self, base_config, config_dict='None'):
        """
        Parameters
        ----------
        run_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """
        if config_dict == "None":
            print("No config file supplied, using default parameters")
            config_dict = def_param
        config_dict = check_and_print_params(config_dict, def_param,
                                             desc_dict)
        super().__init__(base_config=base_config, config_dict=config_dict)

        inputs = self.config_dict['run_params']

        self.width = inputs['rand_width']
        self.zmin = inputs['rand_zmin']
        self.zmax = inputs['rand_zmax']
        self.nzbins = inputs['nzbins']
        np.random.seed(87)

    def inform(self, training_data):
        """
          this is random, so does nothing
        """
        print("I don't need to train!!!")
        pprint.pprint(training_data)

        pass

    def load_pretrained_model(self):
        pass

    def estimate(self, test_data):
        pdf = []
        # allow for either format for now
        try:
            d = test_data['i_mag']
        except Exception:
            d = test_data['mag_i_lsst']
        numzs = len(d)
        zmode = np.round(np.random.uniform(0.0, self.zmax, numzs), 3)
        widths = self.width * (1.0 + zmode)
        self.zgrid = np.linspace(0., self.zmax, self.nzbins)
        for i in range(numzs):
            pdf.append(norm.pdf(self.zgrid, zmode[i], widths[i]))
        if self.output_format == 'qp':
            qp_d = qp.Ensemble(qp.stats.norm, data=dict(loc=zmode,
                                                        scale=widths))
            return qp_d
        else:
            pz_dict = {'zmode': zmode, 'pz_pdf': pdf}
            return pz_dict
