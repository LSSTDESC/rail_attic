"""
Example code that just spits out random numbers between 0 and 3
for z_mode, and Gaussian centered at z_mode with width
random_width*(1+zmode).
"""

import numpy as np
from scipy.stats import norm
from rail.estimation.estimator import Estimator as BaseEstimation

class randomPZ(BaseEstimation):
   
    def __init__(self, config_dict):
        """
        Parameters:
        -----------
        run_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """
        super().__init__(config_dict)
        
        inputs = self.config_dict['run_params']

        self.width = inputs['rand_width']
        self.zmin = inputs['rand_zmin']
        self.zmax = inputs['rand_zmax']
        self.nzbins = inputs['nzbins']

    def train(self):
        """
          this is random, so does nothing
        """
        print("I don't need to train!!!")
        pass

    def run_photoz(self,test_data):
        print("running photoz's...")
        pdf = []
        numzs = len(test_data['mag_i_lsst'])
        zmode = np.random.uniform(0.0, self.zmax, numzs)
        widths = self.width * (1.0 + zmode)
        self.zgrid = np.linspace(0., self.zmax, 301)
        for i in range(numzs):
            pdf.append(norm.pdf(self.zgrid, zmode[i], widths[i]))
        pz_dict ={'zmode':zmode, 'pz_pdf':pdf}
        return pz_dict
