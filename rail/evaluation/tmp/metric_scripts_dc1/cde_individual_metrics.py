from __future__ import division
import numpy as np
import qp
import matplotlib.pyplot as plt
from scipy import stats
import skgof

class EvaluateMetric(object):
    def __init__(self,ensemble_obj,truths):
        """an object that takes a qp Ensemble of N PDF objects and an array of
        N "truth" specz's, will be used to calculate PIT and QQ, plus more stuff
        later
        Parameters:
        ensemble_obj: qp ensemble 
            a qp ensemble object containing the N PDFs
        truths: numpy array 
            1D numpy array with the N spec-z values
        self.pitarray stores the PIT values once they are computed, as they are 
            used by multiple metrics, so there's no need to compute them twice
        """

#        if ensemble_obj==None or truths==None:
#            print 'Warning: inputs not complete'
        self.ensemble_obj = ensemble_obj
        self.truths = truths
        self.pitarray = None #will store once computed as used often
        

    def PIT(self,using='gridded',dx=0.0001):
        """ computes the Probability Integral Transform (PIT), described in 
        Tanaka et al. 2017(ArXiv 1704.05988), which is the integral of the 
        PDF from 0 to zref.
        Parameters:
        using: string
             which parameterization to evaluate
        dx: float
             step size used in integral
        Returns
        -------
        ndarray
             The values of the PIT for each ensemble object
             Also stores PIT array in self.pitarray
        """
        if len(self.truths) != self.ensemble_obj.n_pdfs:
            print 'Warning: number of zref values not equal to number of ensemble objects'
            return
        n = self.ensemble_obj.n_pdfs
        pitlimits = np.zeros([n,2])
        pitlimits[:,1] = self.truths
        tmppit = self.ensemble_obj.integrate(limits=pitlimits,using=using,dx=dx)
        self.pitarray = np.array(tmppit)
        return tmppit

    def QQvectors(self,using,dx=0.0001,Nquants=101):
        """Returns quantile quantile vectors for the ensemble using the PIT values,
        without actually plotting them.  Will be useful in making multi-panel plots 
        simply take the percentiles of the values in order to get the Qdata
        quantiles
        Parameters:
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        Nquants: int
            the number of quantile bins to compute, default 100
        Returns
        -------
        numpy arrays for Qtheory and Qdata
        """
        if self.pitarray is not None:
            pits = np.array(self.pitarray)
        else:
            pits = self.PIT(using=using,dx=dx)
            self.pitarray = pits
        quants = np.linspace(0.,100.,Nquants)
        Qtheory = quants/100.
        Qdata = np.percentile(pits,quants)
        return Qtheory, Qdata

    def QQplot(self,using,dx=0.0001,Nquants=101):
        """Quantile quantile plot for the ensemble using the PIT values, 
        simply take the percentiles of the values in order to get the Qdata
        quantiles
        Parameters:
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        Nquants: int
            the number of quantile bins to compute, default 100
        Returns
        -------
        matplotlib plot of the quantiles
        """
        if self.pitarray is not None:
            pits = np.array(self.pitarray)
        else:
            pits = self.PIT(using=using,dx=dx)
            self.pitarray = pits
        quants = np.linspace(0.,100.,Nquants)
        QTheory = quants/100.
        Qdata = np.percentile(pits,quants)
        plt.figure(figsize=(10,10))
        plt.plot(QTheory,Qdata,c='b',linestyle='-',linewidth=3,label='QQ')
        plt.plot([0,1],[0,1],color='k',linestyle='-',linewidth=2)
        plt.xlabel("Qtheory",fontsize=18)
        plt.ylabel("Qdata",fontsize=18)
        plt.legend()
        plt.savefig("QQplot.jpg")
        return


    def KS(self, using, dx=0.0001):
        """
        Compute the Kolmogorov-Smirnov statistic and p-value for the PIT 
        values by comparing with a uniform distribution between 0 and 1. 
        Parameters:
        -----------
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        Returns:
        --------
        KS statistic and pvalue

        """
        if self.pitarray is not None:
            pits = np.array(self.pitarray)
        else:
            pits = np.array(self.PIT(using=using,dx=dx))
            self.pitarray = pits
        ks_result = skgof.ks_test(pits, stats.uniform())
        return ks_result.statistic, ks_result.pvalue

    def CvM(self, using, dx=0.0001):
        """
        Compute the Cramer-von Mises statistic and p-value for the PIT values
        by comparing with a uniform distribution between 0 and 1. 
        Parameters:
        -----------
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        Returns:
        --------
        CvM statistic and pvalue

        """
        if self.pitarray is not None:
            pits = np.array(self.pitarray)
        else:
            pits = np.array(self.PIT(using=using,dx=dx))
            self.pitarray = pits
        cvm_result = skgof.cvm_test(pits, stats.uniform())
        return cvm_result.statistic, cvm_result.pvalue

    def AD(self, using, dx=0.0001, vmin=0.005, vmax=0.995):
        """
        Compute the Anderson-Darling statistic and p-value for the PIT 
        values by comparing with a uniform distribution between 0 and 1. 
        
        Since the statistic diverges at 0 and 1, PIT values too close to
        0 or 1 are discarded. 

        Parameters:
        -----------
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        vmin, vmax: floats
            PIT values outside this range are discarded
        Returns:
        --------
        AD statistic and pvalue

        """
        if self.pitarray is not None:
            pits = np.array(self.pitarray)
        else:
            pits = np.array(self.PIT(using=using,dx=dx))
            self.pitarray = pits
        mask = (pits>vmin) & (pits<vmax)
        print "now with proper uniform range"
        delv = vmax-vmin
        ad_result = skgof.ad_test(pits[mask], stats.uniform(loc=vmin,scale=delv))
        return ad_result.statistic, ad_result.pvalue

    def cde_loss(self, grid):
        """Computes the estimated conditional density loss described in
        Izbicki & Lee 2017 (arXiv:1704.08095).

        Parameters:
        grid: np array of values at which to evaluate the pdf.
        Returns:
        an estimate of the cde loss.
        """
        grid, pdfs = self.ensemble_obj.evaluate(grid, norm=True)

        n_obs, n_grid = pdfs.shape

        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(pdfs ** 2, grid))

        # Calculate second term E[f*(Z | X)]
        nns = [np.argmin(np.abs(grid - true_z)) for true_z in self.truths]
        term2 = np.mean(pdfs[range(n_obs), nns])

        return term1 - 2 * term2


class NzSumEvaluateMetric(object):
    def __init__(self,ensemble_obj,truth_vals,eval_grid=None,using='gridded',dx=0.001):
        """an object that takes a qp Ensemble of N PDF objects and an array of
        N "truth" specz's, will be used to calculate PIT and QQ, plus more stuff
        later
        Parameters:
        ensemble_obj: qp ensemble object 
            a qp ensemble object of N PDFs that will be stacked
        truths: numpy array of N true spec-z values
            1D numpy array with the N spec-z values
        eval_grid: the numpy array to evaluate the metrics on.  If fed "None"
            will default to np.arange(0.005,2.12,0.01), i.e. the grid for BPZ
        """

#        if stackpz_obj==None or truth_vals==None:
#            print 'Warning: inputs not complete'
        self.ensemble_obj = ensemble_obj
        self.truth = truth_vals
        if eval_grid is None:
            self.eval_grid = np.arange(0.005,2.12,0.01)
            print "using default evaluation grid of numpy.arange(0.005,2.12,0.01)\n"
        else: 
            self.eval_grid = eval_grid
        self.using=using
        self.dx = dx
        
        #make a stack of the ensemble object evaluated at the eval_grid points
        stacked = self.ensemble_obj.stack(loc=self.eval_grid,using='gridded')
        self.stackpz = qp.PDF(gridded=(stacked['gridded'][0],stacked['gridded'][1]))
        return
       

    def NZKS(self):
        """
        Compute the Kolmogorov-Smirnov statistic and p-value for the 
        two distributions of sumpz and true_z
        Parameters:
        -----------
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        Returns:
        --------
        KS statistic and pvalue

        """
        #copy the form of Rongpu's use of skgof functions
        #will have to use QPPDFCDF class, as those expect objects
        #that have a .cdf method for a vector of values
        tmpnzfunc = QPPDFCDF(self.stackpz)
        nzks = skgof.ks_test(self.truth,tmpnzfunc)
        return nzks.statistic, nzks.pvalue
        

#        #copy Kartheik's metric qp.PDF.evaluate returns the array points as
#        #[0] and the values as [1], so pull out just the values
#        p1obs = self.stackpz.evaluate(self.eval_grid,using=using, vb=False)[1]
#        p2truth = self.truth.evaluate(self.eval_grid,using=using, vb=False)[1]
#       
#        ks_stat, ks_pval = stats.ks_2samp(p1obs,p2truth)
#
#        return ks_stat, ks_pval

    def NZCVM(self):
      """                                                                              
      Compute the CramervonMises statistic and p-value for the
      two distributions of sumpz and true_z vector of spec-z's
      Parameters:                                                                      
      -----------
      using: string
      which parameterization to evaluate
      Returns:
      --------
      CvM statistic and pvalue
      """
      #copy the form of Rongpu's use of skgof functions
      #will have to use QPPDFCDF class, as those expect objects
      #that have a .cdf method for a vector of values
      tmpnzfunc = QPPDFCDF(self.stackpz)
      nzCvM = skgof.cvm_test(self.truth,tmpnzfunc)
      return nzCvM.statistic, nzCvM.pvalue

    def NZAD(self, vmin = 0.005, vmax = 1.995):
      """                                                                              
      Compute the Anderson Darling statistic and p-value for the
      two distributions of sumpz and true_z vector of spec-z's
      Parameters:                                      
      vmin, vmax: specz values outside of these values are discarded
      -----------
      using: string
      which parameterization to evaluate
      Returns:
      --------
      Anderson-Darling statistic and pvalue
      """
      #copy the form of Rongpu's use of skgof functions
      #will have to use QPPDFCDF class, as those expect objects
      #that have a .cdf method for a vector of values
      print "using %f and %f for vmin and vmax\n"%(vmin,vmax)
      szs = self.truth
      mask = (szs > vmin) & (szs < vmax)

      tmpnzfunc = QPPDFCDF(self.stackpz,self.dx)
      nzAD = skgof.ad_test(szs[mask],tmpnzfunc)
      return nzAD.statistic, nzAD.pvalue




class QPPDFCDF(object):
    def __init__(self,pdf_obj,dx=0.0001):
        """an quick wrapper for a qp.PDF object that has .pdf and .cdf 
        functions for use with skgof functions
        pdf_obj: qp pdf object with using='gridded' parameterization
        """

        self.pdf_obj = pdf_obj
        self.dx = dx
        return

    def pdf(self,grid):
        """
        returns pdf of qp.PDF object by calling qp.PDF.evaluate
        inputs:
            grid:: float or np ndarray of values to evaluate the pdf at
        returns:
            pdf of object evaluated at points in grid
        """
        return self.pdf_obj.evaluate(grid,'gridded',False,False)[1]

    def cdf(self,xvals):
        """
        returns CDF of qp.PDF object by calling qp.PDF.integrate
        with limits between 0.0 and loc
        inputs: 
            vals: float or np ndarray of values to evaluate the pdf at
        Returns:
            the array of cdf values evaluated at vals
        """
        vals = np.array(xvals)
        if vals.size ==1:
            lims = (0.0,xvals)
            cdfs = self.pdf_obj.integrate(lims,self.dx,'gridded',False)
        else:
            nval = len(vals)
            cdfs = np.zeros(nval)
            for i in range(nval):
                lims = (0.0,vals[i])
                cdfs[i] = self.pdf_obj.integrate(lims,self.dx,'gridded',False)
        return cdfs
