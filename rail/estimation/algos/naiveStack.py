"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.summarizer import PZSummarizer
from rail.core.data import QPHandle
import qp


class NaiveStack(PZSummarizer):
    """Summarizer which simply histograms a point estimate
    """

    name = 'NaiveStack'
    config_options = PZSummarizer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          seed=Param(int, 87, msg="random seed"),
                          nsamples=Param(int, 1000, msg="Number of sample distributions to create"))
    outputs = [('output', QPHandle),
               ('single_NZ', QPHandle)]

    def __init__(self, args, comm=None):
        PZSummarizer.__init__(self, args, comm=comm)
        self.zgrid = None

    def run(self):
        rng = np.random.default_rng(seed=self.config.seed)
        test_data = self.get_data('input')
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins + 1)
        pdf_vals = test_data.pdf(self.zgrid)
        yvals = np.expand_dims(np.sum(np.where(np.isfinite(pdf_vals), pdf_vals, 0.), axis=0), 0)
        qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=yvals))

        bvals = np.empty((self.config.nsamples, len(self.zgrid)))
        for i in range(self.config.nsamples):
            bootstrap_draws = rng.integers(low=0, high=test_data.npdf, size=test_data.npdf)
            bvals[i] = np.sum(pdf_vals[bootstrap_draws], axis=0)
        sample_ens = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=bvals))

        self.add_data('output', sample_ens)
        self.add_data('single_NZ', qp_d)
