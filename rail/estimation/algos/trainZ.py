"""
Implementation of the 'pathological photo-z PDF estimator,
as used in arXiv:2001.03621 (see section 3.3). It assigns each test set galaxy
a photo-z PDF equal to the normalized redshift distribution
N (z) of the training set.
"""


import pickle
import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.estimator import Estimator, Trainer
import qp


class trainZmodel:
    """
    Temporary class to store the single trainZ pdf for trained model.
    Given how simple this is to compute, this seems like overkill.
    """
    def __init__(self, zgrid, pdf, zmode):
        self.zgrid = zgrid
        self.pdf = pdf
        self.zmode = zmode


class Train_trainZ(Trainer):

    name = 'Train_trainZ'
    config_options = Trainer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"))

    
    def __init__(self, args, comm=None):
        Trainer.__init__(self, args, comm=comm)
        
    def run(self):
        training_data = self.get_data('input')               
        zbins = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins+1)
        speczs = np.sort(training_data['redshift'])
        train_pdf, _ = np.histogram(speczs, zbins)
        midpoints = zbins[:-1] + np.diff(zbins)/2
        self.zmode = midpoints[np.argmax(train_pdf)]
        cdf = np.cumsum(train_pdf)
        self.cdf = cdf / cdf[-1]
        norm = cdf[-1]*(zbins[2]-zbins[1])
        self.train_pdf = train_pdf/norm
        self.zgrid = midpoints
        model = trainZmodel(self.zgrid, self.train_pdf, self.zmode)
        if self.config.save_train:
            with open(self.config.model_file, 'wb') as f:
                pickle.dump(file=f, obj=model,
                            protocol=pickle.HIGHEST_PROTOCOL)
        np.random.seed(87) # set here for tests matching


class TrainZ(Estimator):

    name = 'TrainZ'
    config_options = Estimator.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"))
    
    def __init__(self, args, comm=None):
        Estimator.__init__(self, args, comm=comm)
        
    def open_model(self, **kwargs):
        Estimator.open_model(self, **kwargs)
        self.zgrid = self.model.zgrid
        self.train_pdf = self.model.pdf
        self.zmode = self.model.zmode

    def run(self):
        test_data = self.get_data('input')
        test_size = len(test_data['mag_i_lsst'])
        zmode = np.repeat(self.zmode, test_size)
        qp_d = qp.Ensemble(qp.interp,
                           data=dict(xvals=self.zgrid, yvals=np.tile(self.train_pdf, (test_size, 1))))
        self.add_data('output', qp_d)

