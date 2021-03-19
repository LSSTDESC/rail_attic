"""
Example code that just spits out random numbers between 0 and 3
for z_mode, and Gaussian centered at z_mode with width
random_width*(1+zmode).
"""

import numpy as np
from scipy.stats import norm
from rail.estimation.estimator import Estimator as BaseEstimation
import qp


class randomPZ(BaseEstimation):

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

        self.width = inputs['rand_width']
        self.zmin = inputs['rand_zmin']
        self.zmax = inputs['rand_zmax']
        self.nzbins = inputs['nzbins']
        np.random.seed(87)

    def inform(self):
        """
          this is random, so does nothing
        """
        print("I don't need to train!!!")
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
