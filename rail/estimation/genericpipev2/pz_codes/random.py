"""
Example code that just spits out random numbers between 0 and 3
for z_mode, and Gaussian centered at z_mode with width
random_width*(1+zmode).
"""

import numpy as np
import genericpipev2
from scipy.stats import norm
from .tomo import Tomographer
from .base import BaseEstimation

class randomPZ(Tomographer,BaseEstimation):
#class randomPZ(Tomographer): 
   
    def __init__(self,base_dict):
        """
        Parameters:
        -----------
        run_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """
        inputs = base_dict['run_params']

        self.width = inputs['rand_width']
        self.zmin = inputs['rand_zmin']
        self.zmax = inputs['rand_zmax']
        self.nzbins = inputs['rand_zbins']
        super().__init__(base_dict)

    def train(self):
        """
          this is random, so does nothing
        """
        print("I don't need to train!!!")
        pass

    def run_photoz(self):
        print("running photoz's...")
        pdf = []
        numzs = len(self.test_data['i_mag'])
        self.zmode = np.random.uniform(0.0,self.zmax,numzs)
        widths = self.width*(1.0+self.zmode)
        self.zgrid = np.linspace(0.,self.zmax,301)
        for i in range(numzs):
            pdf.append(norm.pdf(self.zgrid,self.zmode[i],widths[i]))
        self.pz_pdf = pdf
