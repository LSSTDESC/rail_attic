"""
Example code that just spits out random numbers between 0 and 3
for z_mode, and Gaussian centered at z_mode with width
random_width*(1+zmode).
"""

import numpy as np
from scipy.stats import norm
from ceci.config import StageParameter as Param
from rail.estimation.estimator import Estimator
import qp


class RandomPZ(Estimator):
    """Random Estimator
    """

    name = 'RandomPZ'

    config_options = Estimator.config_options.copy()
    config_options.update(rand_width=Param(float, 0.025, "ad hock width of PDF"),
                          rand_zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          rand_zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"))

    def __init__(self, args, comm=None):
        """ Constructor:
        Do Estimator specific initialization """
        Estimator.__init__(self, args, comm=comm)
        self.zgrid = None

    def run(self):

        if self.config.hdf5_groupname:
            test_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  #pragma:  no cover
            test_data = self.get_data('input')
        pdf = []
        # allow for either format for now
        try:
            d = test_data['i_mag']
        except Exception:
            d = test_data['mag_i_lsst']
        numzs = len(d)
        zmode = np.round(np.random.uniform(0.0, self.config.rand_zmax, numzs), 3)
        widths = self.config.rand_width * (1.0 + zmode)
        self.zgrid = np.linspace(self.config.rand_zmin, self.config.rand_zmax, self.config.nzbins)
        for i in range(numzs):
            pdf.append(norm.pdf(self.zgrid, zmode[i], widths[i]))
        qp_d = qp.Ensemble(qp.stats.norm, data=dict(loc=np.expand_dims(zmode, -1),  #pylint: disable=no-member
                                                    scale=np.expand_dims(widths, -1)))
        qp_d.set_ancil(dict(zmode=zmode))
        self.add_data('output', qp_d)
