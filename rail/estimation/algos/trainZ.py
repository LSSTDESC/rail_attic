"""
Implementation of the 'pathological photo-z PDF estimator,
as used in arXiv:2001.03621 (see section 3.3). It assigns each test set galaxy
a photo-z PDF equal to the normalized redshift distribution
N (z) of the training set.
"""


import pickle
import numpy as np
from rail.estimation.estimator import Estimator as BaseEstimation
import qp


class trainZ(BaseEstimation):

    def __init__(self, base_config, config_dict):
        super().__init__(base_config=base_config, config_dict=config_dict)

        inputs = self.config_dict['run_params']
        self.zmin = inputs['zmin']
        self.zmax = inputs['zmax']
        self.nzbins = inputs['nzbins']
        self.inform_options = inputs['inform_options']

    def inform(self, training_data):
        zbins = np.linspace(self.zmin, self.zmax, self.nzbins+1)
        speczs = np.sort(training_data['redshift'])
        train_pdf, _ = np.histogram(speczs, zbins)
        midpoints = zbins[:-1] + np.diff(zbins)/2
        self.zmode = midpoints[np.argmax(train_pdf)]
        cdf = np.cumsum(train_pdf)
        self.cdf = cdf / cdf[-1]
        self.train_pdf = train_pdf/self.cdf
        self.zgrid = midpoints
        model = trainZmodel(self.zgrid, self.train_pdf, self.zmode)
        if self.inform_options['save_train']:
            with open(self.inform_options['modelfile'], 'wb') as f:
                pickle.dump(file=f, obj=model,
                            protocol=pickle.HIGHEST_PROTOCOL)
        np.random.seed(87) # set here for tests matching

    def load_pretrained_model(self):
        modelfile = self.inform_options['modelfile']
        model = pickle.load(open(modelfile, 'rb'))
        self.zgrid = model.zgrid
        self.train_pdf = model.pdf
        self.zmode = model.zmode

    def estimate(self, test_data):
        test_size = len(test_data['id'])
        zmode = np.repeat(self.zmode, test_size)
        if self.output_format == 'qp':
            qp_d = qp.Ensemble(qp.interp,
                               data=dict(xvals=self.zgrid,
                                         yvals=np.tile(self.train_pdf,
                                                       (test_size, 1))))
            return qp_d
        else:
            pz_dict = {'zmode': zmode, 'pz_pdf': np.tile(self.train_pdf,
                                                         (test_size, 1))}
            return pz_dict


class trainZmodel:
    """
    Temporary class to store the single trainZ pdf for trained model.
    Given how simple this is to compute, this seems like overkill.
    """
    def __init__(self, zgrid, pdf, zmode):
        self.zgrid = zgrid
        self.pdf = pdf
        self.zmode = zmode
