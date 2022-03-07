"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.summarization.summarizer import PZtoNZSummarizer
import qp
from scipy.special import digamma
from scipy.stats import dirichlet

TEENY = 1.e-15

class VarInferenceStack(PZtoNZSummarizer):
    """Variational inference summarizer based on notebook created by Markus Rau
    The summzarizer is appropriate for the likelihoods returned by
    template-based codes, for which the NaiveSummarizer are not appropriate.
    Parameters:
    -----------
    zmin: float
      minimum z for redshift grid
    zmax: float
      maximum z for redshift grid
    nzbins: int
      number of bins for redshift grid
    niter: int
      number of iterations to perform in the variational inference
    nsamples: int
      number of samples used in dirichlet to determind error bar
    """

    name = 'VarInferenceStack'
    config_options = PZtoNZSummarizer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          niter=Param(int, 100, msg="The number of iterations in the variational inference"),
                          nsamples=Param(int, 500, msg="The number of samples used in dirichlet uncertainty"))

    def __init__(self, args, comm=None):
        PZtoNZSummarizer.__init__(self, args, comm=comm)
        self.zgrid = None

    def run(self):
        test_data = self.get_data('input')
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        pdf_vals = test_data.pdf(self.zgrid)
        log_pdf_vals = np.log(np.array(pdf_vals)+TEENY)
        alpha_trace = np.ones(len(self.zgrid))
        init_trace = np.ones(len(self.zgrid))

        for _ in range(self.config.niter):
            dig = np.array([digamma(kk) - digamma(np.sum(alpha_trace)) for kk in alpha_trace])
            matrix_grid = np.exp(dig + log_pdf_vals)
            gamma_matrix = np.array([kk/np.sum(kk) for kk in matrix_grid])
            nk = np.sum(gamma_matrix, axis=0)
            alpha_trace = nk + init_trace

        qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=alpha_trace))
        #sample_pz = dirichlet.rvs(alpha_trace, size=self.config.nsamples)
        #siglow = np.expand_dims(np.percentile(sample_pz, 15.87, axis=0), -1).T
        #sighi = np.expand_dims(np.percentile(sample_pz, 84.13, axis=0), -1).T
        #ancil_dict = dict(sigmalow=siglow, sigmahigh=sighi)
        #qp_d.set_ancil(ancil_dict)
        self.add_data('output', qp_d)
