"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.summarizer import PZSummarizer
import qp

class PointEstimateHist(PZSummarizer):
    """Summarizer which simply histograms a point estimate
    """

    name = 'PointEstimateHist'
    config_options = PZSummarizer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          point_estimate=Param(str, 'zmode', msg="Which point estimate to use"),
                          nsamples=Param(int, 1000, msg="Number of sample distributions to return"))

    def __init__(self, args, comm=None):
        PZSummarizer.__init__(self, args, comm=comm)
        self.zgrid = None

    def run(self):
        test_data = self.get_data('input')
        npdf = test_data.npdf
        zb = test_data.ancil['zmode']
        nsamp = self.config.nsamples
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins+1)
        bootstrap_indeces = np.random.randint(npdf,
                                              size=npdf * nsamp).reshape([nsamp, npdf])
        hist_vals = np.empty((0, self.config.nzbins))
        for i in range(nsamp):
            uniq, cnts = np.unique(bootstrap_indeces[i], return_counts=True)
            zarr = np.array([])
            for un, ct in zip(uniq, cnts):
                zarr = np.concatenate((zarr, np.repeat(zb[un], ct)), axis=None)
            tmp_hist_vals = np.histogram(zarr, bins=self.zgrid)[0]
            hist_vals = np.vstack((hist_vals, tmp_hist_vals))
        qp_d = qp.Ensemble(qp.hist,
                           data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_vals)))
        self.add_data('output', qp_d)
