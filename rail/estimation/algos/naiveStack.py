"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.summarizer import PZSummarizer
import qp

class NaiveStack(PZSummarizer):
    """Summarizer which simply histograms a point estimate
    """

    name = 'NaiveStack'
    config_options = PZSummarizer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"))

    def __init__(self, args, comm=None):
        PZSummarizer.__init__(self, args, comm=comm)
        self.zgrid = None

    def run(self):
        test_data = self.get_data('input')
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins+1)
        pdf_vals = test_data.pdf(self.zgrid)
        yvals = np.expand_dims(np.sum(np.where(np.isfinite(pdf_vals), pdf_vals, 0.), axis=0), 0)
        qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=yvals))
        self.add_data('output', qp_d)
