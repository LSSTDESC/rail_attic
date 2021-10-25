####################################################################################################
# Script name : processFilters.py
#
# fit the band filters with a gaussian mixture
# if make_plot, save images
#
# output file : band + '_gaussian_coefficients.txt'
#####################################################################################################
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import leastsq

from delight.utils import *
from delight.io import *
#from rail.estimation.algos.include_delightPZ.delight_io import *

import coloredlogs
import logging

# Create a logger object.
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s %(name)s[%(process)d] %(levelname)s %(message)s')


def processFilters(configfilename):
    """
    processFilters(configfilename)

    Develop filter transmission functions as a Gaussian Kernel regression

    : input file : the configuration file
    :return:
    """

    msg="----- processFilters ------"
    logger.info(msg)

    
    msg=f"parameter file is {configfilename}"
    logger.info(msg)
        

    params = parseParamFile(configfilename, verbose=False, catFilesNeeded=False)
    


    numCoefs = params["numCoefs"]
    bandNames = params['bandNames']
    make_plots= params['bands_makeplots']

    fmt = '.res'
    max_redshift = params['redshiftMax']  # for plotting purposes
    root = params['bands_directory']

    if make_plots:
        import matplotlib.pyplot as plt
        cm = plt.get_cmap('brg')
        num = len(bandNames)
        cols = [cm(i/num) for i in range(num)]


    # Function we will optimize
    # Gaussian function representing filter
    def dfunc(p, x, yd):
        y = 0*x
        n = p.size//2
        for i in range(n):
            y += np.abs(p[i]) * np.exp(-0.5*((mus[i]-x)/np.abs(p[n+i]))**2.0)
        return yd - y

    if make_plots:
        fig0, ax0 = plt.subplots(1, 1, figsize=(8.2, 4))

    # Loop over bands
    for iband, band in enumerate(bandNames):

        fname_in = root + '/' + band + fmt
        data = np.genfromtxt(fname_in)
        coefs = np.zeros((numCoefs, 3))
        # wavelength - transmission function
        x, y = data[:, 0], data[:, 1]
        #y /= x  # divide by lambda
        # Only consider range where >1% max
        ind = np.where(y > 0.01*np.max(y))[0]
        lambdaMin, lambdaMax = x[ind[0]], x[ind[-1]]

        # Initialize values for amplitude and width of the components
        sig0 = np.repeat((lambdaMax-lambdaMin)/numCoefs/4, numCoefs)
        # Components uniformly distributed in the range
        mus = np.linspace(lambdaMin+sig0[0], lambdaMax-sig0[-1], num=numCoefs)
        amp0 = interp1d(x, y)(mus)
        p0 = np.concatenate((amp0, sig0))
        print(band, end=" ")

        # fit
        popt, pcov = leastsq(dfunc, p0, args=(x, y))
        coefs[:, 0] = np.abs(popt[0:numCoefs])  # amplitudes
        coefs[:, 1] = mus  # positions
        coefs[:, 2] = np.abs(popt[numCoefs:2*numCoefs])  # widths

        # output for gaussian regression fit coefficients
        fname_out = root + '/' + band + '_gaussian_coefficients.txt'
        np.savetxt(fname_out, coefs, header=fname_in)

        xf = np.linspace(lambdaMin, lambdaMax, num=1000)
        yy = 0*xf
        for i in range(numCoefs):
            yy += coefs[i, 0] * np.exp(-0.5*((coefs[i, 1] - xf)/coefs[i, 2])**2.0)

        if make_plots:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x[ind], y[ind], lw=3, label='True filter', c='k')
            ax.plot(xf, yy, lw=2, c='r', label='Gaussian fit')
            # ax0.plot(x[ind], y[ind], lw=3, label=band, color=cols[iband])
            ax0.plot(xf, yy, lw=3, label=band, color=cols[iband])

        coefs_redshifted = 1*coefs
        coefs_redshifted[:, 1] /= (1. + max_redshift)
        coefs_redshifted[:, 2] /= (1. + max_redshift)
        lambdaMin_redshifted, lambdaMax_redshifted\
            = lambdaMin / (1. + max_redshift), lambdaMax / (1. + max_redshift)
        xf = np.linspace(lambdaMin_redshifted, lambdaMax_redshifted, num=1000)
        yy = 0*xf
        for i in range(numCoefs):
            yy += coefs_redshifted[i, 0] *\
                np.exp(-0.5*((coefs_redshifted[i, 1] - xf) /
                       coefs_redshifted[i, 2])**2.0)

        if make_plots:
            ax.plot(xf, yy, lw=2, c='b', label='G fit at z='+str(max_redshift))
            title = band + ' band (' + fname_in +\
                ') with %i' % numCoefs+' components'
            ax.set_title(title)
            ax.set_ylim([0, data[:, 1].max()*1.2])
            ax.set_yticks([])
            ax.set_xlabel('$\lambda$')
            ax.legend(loc='upper center', frameon=False, ncol=3)

            fig.tight_layout()
            fname_fig = root + '/' + band + '_gaussian_approximation.png'
            fig.savefig(fname_fig)

    if make_plots:
        ax0.legend(loc='upper center', frameon=False, ncol=4)
        ylims = ax0.get_ylim()
        ax0.set_ylim([0, 1.4*ylims[1]])
        ax0.set_yticks([])
        ax0.set_xlabel(r'$\lambda$')
        fig0.tight_layout()
        fname_fig = root + '/allbands.pdf'
        fig0.savefig(fname_fig)



#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # execute only if run as a script


    msg="Start processFilters.py"
    logger.info(msg)
    logger.info("--- Process FILTERS ---")

    #numCoefs = 7  # number of components for the fit
    #numCoefs = 21  # for lsst the transmission is too wavy ,number of components for the fit
    #make_plots = True

    if len(sys.argv) < 2:
        raise Exception('Please provide a parameter file')

    processFilters(sys.argv[1])
