#######################################################################################################
#
# script : simulateWithSED.py
#
# simulate mock data with those filters and SEDs
# produce files `galaxies-redshiftpdfs.txt` and `galaxies-redshiftpdfs2.txt` for training and target
#
#########################################################################################################


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from delight.io import *
from delight.utils import *


import coloredlogs
import logging


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s, %(name)s[%(process)d] %(levelname)s %(message)s')


def simulateWithSEDs(configfilename):
    """

    :param configfilename:
    :return:
    """




    logger.info("--- Simulate with SED ---")

    params = parseParamFile(configfilename, verbose=False, catFilesNeeded=False)
    dir_seds = params['templates_directory']
    sed_names = params['templates_names']

    # redshift grid
    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)

    numZ = redshiftGrid.size
    numT = len(sed_names)
    numB = len(params['bandNames'])
    numObjects = params['numObjects']
    noiseLevel = params['noiseLevel']

    # f_mod : 2D-container of interpolation functions of flux over redshift:
    # row sed, column bands
    # one row per sed, one column per band
    f_mod = np.zeros((numT, numB), dtype=object)

    # loop on SED
    # read the fluxes file at different redshift in training data file
    # in file sed_name + '_fluxredshiftmod.txt'
    # to produce f_mod the interpolation function redshift --> flux for each band and sed template
    for it, sed_name in enumerate(sed_names):
        # data : redshifted fluxes (row vary with z, columns: filters)
        data = np.loadtxt(dir_seds + '/' + sed_name + '_fluxredshiftmod.txt')
        # build the interpolation of flux wrt redshift for each band
        for jf in range(numB):
            f_mod[it, jf] = interp1d(redshiftGrid, data[:, jf], kind='linear')

    # Generate training data
    #-------------------------
    # pick a set of redshift at random to be representative of training galaxies
    redshifts = np.random.uniform(low=redshiftGrid[0],high=redshiftGrid[-1],size=numObjects)
    #pick some SED type at random
    types = np.random.randint(0, high=numT, size=numObjects)

    ell = 1e6 # I don't know why we have this value multiplicative constant
              # it is to show that delightLearn can find this multiplicative number when calling
              # utils:scalefree_flux_likelihood(returnedChi2=True)
    #ell = 0.45e-4 # SDC may 14 2021 calibrate approximately to AB magnitude

    # what is fluxes and fluxes variance
    fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))

    # loop on objects to simulate for the training and save in output training file
    for k in range(numObjects):
        #loop on number of bands
        for i in range(numB):
            trueFlux = ell * f_mod[types[k], i](redshifts[k]) # noiseless flux at the random redshift
            noise = trueFlux * noiseLevel
            fluxes[k, i] = trueFlux + noise * np.random.randn() # noisy flux
            fluxesVar[k, i] = noise**2.

    # container for training galaxies output
    # at some redshift, provides the flux and its variance inside each band
    data = np.zeros((numObjects, 1 + len(params['training_bandOrder'])))
    bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,refBandColumn = readColumnPositions(params, prefix="training_")

    for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
        data[:, pf] = fluxes[:, ib]
        data[:, pfv] = fluxesVar[:, ib]
    data[:, redshiftColumn] = redshifts
    data[:, -1] = types
    np.savetxt(params['trainingFile'], data)

    # Generate Target data : procedure similar to the training
    #-----------------------------------------------------------
    # pick set of redshift at random
    redshifts = np.random.uniform(low=redshiftGrid[0],high=redshiftGrid[-1],size=numObjects)
    types = np.random.randint(0, high=numT, size=numObjects)

    fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))

    # loop on objects in target files
    for k in range(numObjects):
        # loop on bands
        for i in range(numB):
            # compute the flux in that band at the redshift
            trueFlux = f_mod[types[k], i](redshifts[k])
            noise = trueFlux * noiseLevel
            fluxes[k, i] = trueFlux + noise * np.random.randn()
            fluxesVar[k, i] = noise**2.

    data = np.zeros((numObjects, 1 + len(params['target_bandOrder'])))
    bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,refBandColumn = readColumnPositions(params, prefix="target_")

    for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
        data[:, pf] = fluxes[:, ib]
        data[:, pfv] = fluxesVar[:, ib]
    data[:, redshiftColumn] = redshifts
    data[:, -1] = types
    np.savetxt(params['targetFile'], data)


if __name__ == "__main__":
    # execute only if run as a script


    msg="Start simulateWithSEDs.py"
    logger.info(msg)
    logger.info("--- simulate with SED ---")



    if len(sys.argv) < 2:
        raise Exception('Please provide a parameter file')

    simulateWithSEDs(sys.argv[1])
