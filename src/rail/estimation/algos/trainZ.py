"""
Implementation of the 'pathological photo-z PDF estimator,
as used in arXiv:2001.03621 (see section 3.3). It assigns each test set galaxy
a photo-z PDF equal to the normalized redshift distribution
N (z) of the training set.
"""

import numpy as np
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.common_params import SHARED_PARAMS
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


class Inform_trainZ(CatInformer):
    """Train an Estimator which returns a global PDF for all galaxies
    """

    name = 'Inform_trainZ'
    config_options = CatInformer.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS)


    def __init__(self, args, comm=None):
        CatInformer.__init__(self, args, comm=comm)

    def run(self):
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  #pragma: no cover
            training_data = self.get_data('input')
        zbins = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins+1)
        speczs = np.sort(training_data['redshift'])
        train_pdf, _ = np.histogram(speczs, zbins)
        midpoints = zbins[:-1] + np.diff(zbins)/2
        zmode = midpoints[np.argmax(train_pdf)]
        cdf = np.cumsum(train_pdf)
        cdf = cdf / cdf[-1]
        norm = cdf[-1]*(zbins[2]-zbins[1])
        train_pdf = train_pdf/norm
        zgrid = midpoints
        self.model = trainZmodel(zgrid, train_pdf, zmode)
        self.add_data('model', self.model)


class TrainZ(CatEstimator):
    """CatEstimator which returns a global PDF for all galaxies
    """

    name = 'TrainZ'
    config_options = CatEstimator.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS)

    def __init__(self, args, comm=None):
        self.zgrid = None
        self.train_pdf = None
        self.zmode = None
        CatEstimator.__init__(self, args, comm=comm)

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.model is None:  #pragma: no cover
            return
        self.zgrid = self.model.zgrid
        self.train_pdf = self.model.pdf
        self.zmode = self.model.zmode

    def _process_chunk(self, start, end, data, first):
        test_size = len(data['mag_i_lsst'])
        zmode = np.repeat(self.zmode, test_size)
        qp_d = qp.Ensemble(qp.interp,
                           data=dict(xvals=self.zgrid, yvals=np.tile(self.train_pdf, (test_size, 1))))
        qp_d.set_ancil(dict(zmode=zmode))
        self._do_chunk_output(qp_d, start, end, first)
