"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.summarizer import PZSummarizer
from rail.core.data import QPHandle
import qp


class PointEstimateHist(PZSummarizer):
    """Summarizer which simply histograms a point estimate
    """

    name = 'PointEstimateHist'
    config_options = PZSummarizer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          seed=Param(int, 87, msg="random seed"),
                          point_estimate=Param(str, 'zmode', msg="Which point estimate to use"),
                          nsamples=Param(int, 1000, msg="Number of sample distributions to return"))
    outputs = [('output', QPHandle),
               ('single_NZ', QPHandle)]

    def __init__(self, args, comm=None):
        PZSummarizer.__init__(self, args, comm=comm)
        self.zgrid = None
        self.bincents = None

    def run(self):
        rng = np.random.default_rng(seed=self.config.seed)
        test_data = self.get_data('input')
        npdf = test_data.npdf
        zb = test_data.ancil['zmode']
        nsamp = self.config.nsamples
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins + 1)
        self.bincents = 0.5 * (self.zgrid[1:] + self.zgrid[:-1])
        single_hist = np.histogram(test_data.ancil[self.config.point_estimate], bins=self.zgrid)[0]
        qp_d = qp.Ensemble(qp.hist,
                           data=dict(bins=self.zgrid, pdfs=np.atleast_2d(single_hist)))
        hist_vals = np.empty((nsamp, self.config.nzbins))
        for i in range(nsamp):
            bootstrap_indeces = rng.integers(low=0, high=npdf, size=npdf)
            zarr = zb[bootstrap_indeces]
            hist_vals[i] = np.histogram(zarr, bins=self.zgrid)[0]
        sample_ens = qp.Ensemble(qp.hist,
                                 data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_vals)))
        self.add_data('output', sample_ens)
        self.add_data('single_NZ', qp_d)
