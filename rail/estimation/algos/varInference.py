"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.summarizer import PZSummarizer
from rail.core.data import QPHandle
import qp
from scipy.special import digamma
from scipy.stats import dirichlet

TEENY = 1.e-15


class VarInferenceStack(PZSummarizer):
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
    config_options = PZSummarizer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          niter=Param(int, 100, msg="The number of iterations in the variational inference"),
                          nsamples=Param(int, 500, msg="The number of samples used in dirichlet uncertainty"))
    outputs = [('output', QPHandle),
               ('single_NZ', QPHandle)]

    def __init__(self, args, comm=None):
        PZSummarizer.__init__(self, args, comm=comm)
        self.zgrid = None

    def run(self):
        test_data = self.get_data('input')
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        pdf_vals = test_data.pdf(self.zgrid)
        log_pdf_vals = np.log(np.array(pdf_vals) + TEENY)
        alpha_trace = np.ones(len(self.zgrid))
        init_trace = np.ones(len(self.zgrid))

        for _ in range(self.config.niter):
            dig = np.array([digamma(kk) - digamma(np.sum(alpha_trace)) for kk in alpha_trace])
            matrix_grid = np.exp(dig + log_pdf_vals)
            gamma_matrix = np.array([kk / np.sum(kk) for kk in matrix_grid])
            nk = np.sum(gamma_matrix, axis=0)
            alpha_trace = nk + init_trace

        # old way of just spitting out a single distribution
        # qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=alpha_trace))
        # instead, sample and save the samples
        sample_pz = dirichlet.rvs(alpha_trace, size=self.config.nsamples)

        qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=alpha_trace))

        sample_ens = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=sample_pz))
        self.add_data('output', sample_ens)

        # now, compute 1 sigma error and add to the initial N(z) estimate
        pdf_vals = sample_ens.pdf(self.zgrid)
        nz_vals = qp_d.pdf(self.zgrid)
        xlow = np.percentile(pdf_vals, 15.87, axis=0)
        xhigh = np.percentile(pdf_vals, 84.13, axis=0)
        sighigh = np.expand_dims(xhigh - nz_vals, -1).T
        siglow = np.expand_dims(nz_vals - xlow, -1).T
        ancil_dict = dict(sigma_low=siglow, sigma_high=sighigh)
        qp_d.set_ancil(ancil_dict)
        self.add_data('single_NZ', qp_d)
