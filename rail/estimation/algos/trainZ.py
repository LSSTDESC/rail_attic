"""
Implementation of the 'pathological photo-z PDF estimator,
as used in arXiv:2001.03621 (see section 3.3). It assigns each test set galaxy
a photo-z PDF equal to the normalized redshift distribution
N (z) of the training set.
"""


import numpy as np
from rail.estimation.estimator import Estimator as BaseEstimation


class trainZ(BaseEstimation):

    def __init__(self, base_config, config_dict):
        super().__init__(base_config=base_config, config_dict=config_dict)

        inputs = self.config_dict['run_params']
        self.zmin = inputs['zmin']
        self.zmax = inputs['zmax']
        self.nzbins = inputs['nzbins']

        zbins = np.linspace(self.zmin, self.zmax, self.nzbins+1)
        speczs = np.sort(self.training_data['redshift'])
        self.train_pdf, _ = np.histogram(speczs, zbins)
        self.midpoints = zbins[:-1] + np.diff(zbins)/2
        self.zmode = self.midpoints[np.argmax(self.train_pdf)]
        cdf = np.cumsum(self.train_pdf)
        self.cdf = cdf / cdf[-1]
        self.zgrid = self.midpoints
        np.random.seed(87)

    def inform(self):
        pass

    def estimate(self, test_data):
        test_size = len(test_data['id'])
        zmode = np.repeat(self.zmode, test_size)
        pz_dict = {'zmode': zmode, 'pz_pdf': np.tile(self.train_pdf,
                                                     (test_size, 1))}
        return pz_dict
