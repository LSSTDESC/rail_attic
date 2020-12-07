import numpy as np
import qp
import matplotlib.pyplot as plt
from scipy import stats

class EvaluatePointStats(object):
    def __init__(self,pzvec,szvec,magvec,imagcut=25.3):
        """An object that takes in the vectors of the point photo-z
        the spec-z, and the i-band magnitudes for calculating the 
        point statistics
        Parameters:
        pzvec: Numpy 1d array of the point photo-z values
        szvec: Numpy 1d array of the spec-z values
        magvec: Numpy 1d array of the i-band magnitudes
        imagcut: float: i-band magnitude cut for the sample
        Calculates: 
        ez_all: (pz-sz)/(1+sz), the quantity will be useful for calculating statistics
        ez_magcut: ez sample trimmed with imagcut
        """
        self.pzs = pzvec
        self.szs = szvec
        self.mags = magvec
        self.imagcut = imagcut
        ez = (pzvec - szvec)/(1.+szvec)
        self.ez_all = ez
        mask = (magvec<imagcut)
        ezcut = ez[mask]
        self.ez_magcut = ezcut

    def CalculateSigmaIQR(self):
        """Calculate the width of the e_z distribution
        using the Interquartile range
        Parameters:
        imagcut: float: i-band magnitude cut for the sample
        Returns: 
        sigma_IQR_all float: width of ez distribution for full sample
        sigma_IQR_magcut float: width of ez distribution for magcut sample
        """
        x75,x25 = np.percentile(self.ez_all,[75.,25.])
        iqr_all = x75-x25
        sigma_iqr_all = iqr_all/1.349
        self.sigma_iqr_all = sigma_iqr_all

        xx75,xx25 = np.percentile(self.ez_magcut,[75.,25.])
        iqr_cut = xx75 - xx25
        sigma_iqr_cut= iqr_cut/1.349 
        self.sigma_iqr_magcut = sigma_iqr_cut
        #store the sigmas for the catastrophic outlier calculation

        return sigma_iqr_all,sigma_iqr_cut

    def CalculateBias(self):
        """calculates the bias of the ez and ez_magcut samples.  In 
        keeping with the Science Book, this is just the median of the
        ez values
        Returns:
        bias_all: median of the full ez sample
        bias_magcut: median of the magcut ez sample
        """
        bias_all = np.median(self.ez_all)
        bias_magcut = np.median(self.ez_magcut)
        return bias_all,bias_magcut

    def CalculateOutlierRate(self):
        """Calculates the catastrophic outlier rate, defined in the
        Science Book as the number of galaxies with ez larger than
        max(0.06,3sigma).  This keeps the fraction reasonable when
        sigma is very small.
        Returns:
        frac_all: fraction of catastrophic outliers for full sample
        frac_magcut: fraction of catastrophic outliers for magcut 
        sample
        """
        num_all = len(self.ez_all)
        num_magcut = len(self.ez_magcut)
        threesig_all = 3.0*self.sigma_iqr_all
        threesig_cut = 3.0*self.sigma_iqr_magcut
        cutcriterion_all = np.maximum(0.06,threesig_all)
        cutcriterion_magcut = np.maximum(0.06,threesig_cut)
        print "using %.3g for cut for whole sample and %.3g for magcut sample\n"%(cutcriterion_all,cutcriterion_magcut)
        mask_all = (self.ez_all>np.fabs(cutcriterion_all))
        mask_magcut = (self.ez_magcut>np.fabs(cutcriterion_magcut))
        outlier_all = np.sum(mask_all)
        outlier_magcut = np.sum(mask_magcut)
        frac_all = float(outlier_all)/float(num_all)
        frac_magcut = float(outlier_magcut)/float(num_magcut)
        return frac_all,frac_magcut

    def CalculateSigmaMAD(self):
        """Function to calculate median absolute deviation and sigma
        based on MAD (just scaled up by 1.4826) for the full and 
        magnitude trimmed samples of ez values
        Returns:
        sigma_mad_all: sigma_MAD for full sample
        sigma_mad_cut: sigma_MAD for the magnitude cut sample
        """
        tmpmed_all = np.median(self.ez_all)
        tmpmed_cut = np.median(self.ez_magcut)
        tmpx_all = np.fabs(self.ez_all - tmpmed_all)
        tmpx_cut = np.fabs(self.ez_magcut - tmpmed_cut)
        mad_all = np.median(tmpx_all)
        mad_cut = np.median(tmpx_cut)
        sigma_mad_all = mad_all*1.4826
        sigma_mad_cut = mad_cut*1.4826
        return sigma_mad_all, sigma_mad_cut
