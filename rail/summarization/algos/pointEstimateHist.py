"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.summarization.summarizer import PZtoNZSummarizer
import qp

class PointEstimateHist(PZtoNZSummarizer):
    """Summarizer which simply histograms a point estimate
    """

    name = 'PointEstimateHist'
    config_options = PZtoNZSummarizer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          point_estimate=Param(str, 'zmode', msg="Which point estimate to use"))

    def __init__(self, args, comm=None):
        PZtoNZSummarizer.__init__(self, args, comm=comm)
        self.zgrid = None

    def run(self):
        test_data = self.get_data('input')
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins+1)
        hist_vals = np.histogram(test_data.ancil[self.config.point_estimate], bins=self.zgrid)[0]
        qp_d = qp.Ensemble(qp.hist,
                           data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_vals)))
        self.add_data('output', qp_d)
