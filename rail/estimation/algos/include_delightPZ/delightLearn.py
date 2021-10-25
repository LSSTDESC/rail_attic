##################################################################################################################################
#
# script : delight-learn.py
#
#  input : 'training_catFile'
#  output : localData or reducedData usefull for Gaussian Process in 'training_paramFile'
#  - find the normalisation of the flux and the best galaxy type
############################################################################################################################
import sys
from mpi4py import MPI
import numpy as np
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel

import coloredlogs
import logging


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s, %(name)s[%(process)d] %(levelname)s %(message)s')

def delightLearn(configfilename):
    """

    :param configfilename:
    :return:
    """


    comm = MPI.COMM_WORLD
    threadNum = comm.Get_rank()
    numThreads = comm.Get_size()

    #parse arguments

    params = parseParamFile(configfilename, verbose=False)

    if threadNum == 0:
        logger.info("--- DELIGHT-LEARN ---")

    # Read filter coefficients, compute normalization of filters
    bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms = readBandCoefficients(params)
    numBands = bandCoefAmplitudes.shape[0]

    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)

    f_mod = readSEDs(params)

    numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))

    msg= 'Number of Training Objects ' + str(numObjectsTraining)
    logger.info(msg)


    firstLine = int(threadNum * numObjectsTraining / numThreads)
    lastLine = int(min(numObjectsTraining,(threadNum + 1) * numObjectsTraining / numThreads))
    numLines = lastLine - firstLine

    comm.Barrier()
    msg ='Thread ' +  str(threadNum) + ' , analyzes lines ' + str(firstLine) + ' , to ' + str(lastLine)
    logger.info(msg)

    DL = approx_DL()
    gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
              params['lines_pos'], params['lines_width'],
              params['V_C'], params['V_L'],
              params['alpha_C'], params['alpha_L'],
              redshiftGridGP, use_interpolators=True)

    B = numBands
    numCol = 3 + B + B*(B+1)//2 + B + f_mod.shape[0]
    localData = np.zeros((numLines, numCol))
    fmt = '%i ' + '%.12e ' * (localData.shape[1] - 1)

    loc = - 1
    crossValidate = params['training_crossValidate']
    trainingDataIter1 = getDataFromFile(params, firstLine, lastLine,prefix="training_", getXY=True,CV=crossValidate)


    if crossValidate:
        chi2sLocal = None
        bandIndicesCV, bandNamesCV, bandColumnsCV,bandVarColumnsCV, redshiftColumnCV = readColumnPositions(params, prefix="training_CV_", refFlux=False)

    for z, normedRefFlux,\
        bands, fluxes, fluxesVar,\
        bandsCV, fluxesCV, fluxesVarCV,\
        X, Y, Yvar in trainingDataIter1:

        loc += 1

        themod = np.zeros((1, f_mod.shape[0], bands.size))
        for it in range(f_mod.shape[0]):
            for ib, band in enumerate(bands):
                themod[0, it, ib] = f_mod[it, band](z)

        # really calibrate the luminosity parameter l compared to the model
        # according the best type of galaxy
        chi2_grid, ellMLs = scalefree_flux_likelihood(fluxes,fluxesVar,themod,returnChi2=True)

        bestType = np.argmin(chi2_grid)  # best type
        ell = ellMLs[0, bestType]        # the luminosity factor
        X[:, 2] = ell

        gp.setData(X, Y, Yvar, bestType)
        lB = bands.size
        localData[loc, 0] = lB
        localData[loc, 1] = z
        localData[loc, 2] = ell
        localData[loc, 3:3+lB] = bands
        localData[loc, 3+lB:3+f_mod.shape[0]+lB+lB*(lB+1)//2+lB] = gp.getCore()

        if crossValidate:
            model_mean, model_covar = gp.predictAndInterpolate(np.array([z]), ell=ell)
            if chi2sLocal is None:
                chi2sLocal = np.zeros((numObjectsTraining, bandIndicesCV.size))

            ind = np.array([list(bandIndicesCV).index(b) for b in bandsCV])

            chi2sLocal[firstLine + loc, ind] = - 0.5 * (model_mean[0, bandsCV] - fluxesCV)**2 /(model_covar[0, bandsCV] + fluxesVarCV)


    # use MPI to get the totals
    comm.Barrier()
    if threadNum == 0:
        reducedData = np.zeros((numObjectsTraining, numCol))
    else:
        reducedData = None

    if crossValidate:
        chi2sGlobal = np.zeros_like(chi2sLocal)
        comm.Allreduce(chi2sLocal, chi2sGlobal, op=MPI.SUM)
        comm.Barrier()

    firstLines = [int(k*numObjectsTraining/numThreads) for k in range(numThreads)]
    lastLines = [int(min(numObjectsTraining, (k+1)*numObjectsTraining/numThreads)) for k in range(numThreads)]
    sendcounts = tuple([(lastLines[k] - firstLines[k]) * numCol for k in range(numThreads)])
    displacements = tuple([firstLines[k] * numCol for k in range(numThreads)])

    comm.Gatherv(localData, [reducedData, sendcounts, displacements, MPI.DOUBLE])
    comm.Barrier()

    # parameters for the GP process on traniing data are transfered to reduced data and saved in file
    #'training_paramFile'
    if threadNum == 0:
        np.savetxt(params['training_paramFile'], reducedData, fmt=fmt)
        if crossValidate:
            np.savetxt(params['training_CVfile'], chi2sGlobal)


#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # execute only if run as a script


    msg="Start Delight Learn.py"
    logger.info(msg)
    logger.info("--- Process Delight Learn ---")


    if len(sys.argv) < 2:
        raise Exception('Please provide a parameter file')

    delightLearn(sys.argv[1])
