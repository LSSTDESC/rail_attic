import numpy as np
import scipy as sp
from scipy import stats as sps
import sys
import bisect

import matplotlib.pyplot as plt

import qp

class EvaluateMetric(object):

    def __init__(self, pdf_obs = None, pdf_truth = None):

        """
        An object to take in photo-z PDFs from various methods
        and evaluate various metrics of performance.

        Parameters
        ----------
        pdf_obs: a qp object containing a stacked PDF of
            outputs from the photo-z code in consideration
        pdf_truth: a qp object containing truths /spec-z to
            evaluate the metric against. All the parameters of
            the underlying qp object should still work.



        """

        # check if the two objects are proper qp PDF objects.
        # return exception if not

        if pdf_obs == None or pdf_truth == None:
             print 'Warning: initializing with insufficient data'

        if self.pdf_obs is not None:
             self.pdf_obs = pdf_obs
        if self.pdf_truth is not None:
             self.pdf_truth = pdf_truth

        return


    def get_KL_divergence(self,limits=(-10.0,10.0), dx=0.01, vb=True):

        """
        Compute the Kullback-Leibler Divergence from
        the spectroscopic N(z) PDF (truth) to the photo-z PDF estimate

        Parameters
        ----------
        limits: tuple of floats
            endpoints of integration interval in which to calculate KLD
        dx: float
            resolution of integration grid
        vb: boolean
            report on progress to stdout?


        Returns
        -------
        KLD: float
            the value of the Kullback-Leibler Divergence from `q` to `p`

        """

        KLD = qp.utils.calculate_kl_divergence(self.pdf_obs, self.pdf_truth, dx, limits, vb)
        return KLD

    def get_QQ_plot(self,quants=None, percent=10., infty=100., vb=True,num_quantiles = 10,plotflag = True):

        """
        Measure quantiles for the two PDFs and produce a QQ plot, to
        determine whether the two PDFs are drawn from the same underlying
        probability distribution. The plot is essentially a scatter plot
        of the quantiles of the first PDF against the quantiles of the second.

        Parameters
        ----------
        num_quantiles: int
            the number of quantiles to compute.
        plotflag: boolean
            generate QQ plot in addition to returning the arrays
        quants: ndarray, float, optional
            array of quantile locations as decimals
        percent: float, optional
            the separation of the requested quantiles, in percent
        infty: float, optional
            approximate value at which CDF=1.
        vb: boolean
            report on progress to stdout?


        Returns
        -------
        obs_quantiles: ndarray
            set of quantiles computed using the photo-z N(z) estimate
        true_quantiles: ndarray
            set of quantiles computed using the spec-z N(z) sample
        """
        # arguments passed in a messy way, cleanup needed

        obs_quantiles = self.pdf_obs.quantize(N=num_quantiles,quants,percent,infty,vb)
        true_quantiles = self.pdf_truth.quantize(N=num_quantiles,quants,percent,infty,vb)

        if plotflag == True:
            fig = plt.figure(figsize = (6,6))
            plt.scatter(true_quantiles,obs_quantiles)
            plt.xlabel('True N(z) quantiles')
            plt.ylabel('Photo-z N(z) quantiles')
            plt.show()

        return obs_quantiles, true_quantiles


    def get_RMSE(self,limits=(-10.,10.), dx=0.01, vb=False):

        """
        Compute RMSE between the two PDFS

        Parameters
        ----------
        limits: tuple of floats
        endpoints of integration interval in which to calculate RMS
        dx: float
            resolution of integration grid
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        rmse: float
            the value of the RMS error between pdf_obs and pdf_truth
        """

        rmse = qp.utils.calculate_rmse(self.pdf_obs,self.pdf_truth,limits,dx,vb)
        return rmse


    def get_KSdist(self,limits=(-10.,10.), dx=0.01):

        """
        Compute KS distance between the two PDFS

        Parameters
        ----------
        limits: tuple of floats
            endpoints of integration interval in which to calculate RMS
        dx: float
            resolution of integration grid

        Returns
        -------
        ksstat : float
            KS statistic
        ks_pval : float
            two-tailed p-value
        """
        # Make a grid from the limits and resolution
        grid = np.linspace(limits[0], limits[1], npoints)

        # Evaluate the functions on the grid
        p1obs = self.pdf_obs.evaluate(grid, vb=vb)[1]
        p2truth = self.pdf_truth.evaluate(grid, vb=vb)[1]

        ksstat, ks_pval     = scipy.stats.ks_2samp(p1obs,p2truth)
