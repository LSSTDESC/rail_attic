
import sys
from mpi4py import MPI
import numpy as np
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
from delight.utils_cy import approx_flux_likelihood_cy
from time import time

import coloredlogs
import logging


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s, %(name)s[%(process)d] %(levelname)s %(message)s')



def delightApply(configfilename):
    """

    :param configfilename:
    :return:
    """


    comm = MPI.COMM_WORLD
    threadNum = comm.Get_rank()
    numThreads = comm.Get_size()


    params = parseParamFile(configfilename, verbose=False)

    if threadNum == 0:
        #print("--- DELIGHT-APPLY ---")
        logger.info("--- DELIGHT-APPLY ---")


    # Read filter coefficients, compute normalization of filters
    bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms = readBandCoefficients(params)
    numBands = bandCoefAmplitudes.shape[0]

    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
    f_mod_interp = readSEDs(params)
    nt = f_mod_interp.shape[0]
    nz = redshiftGrid.size

    dir_seds = params['templates_directory']
    dir_filters = params['bands_directory']
    lambdaRef = params['lambdaRef']
    sed_names = params['templates_names']
    f_mod_grid = np.zeros((redshiftGrid.size, len(sed_names),len(params['bandNames'])))


    for t, sed_name in enumerate(sed_names):
        f_mod_grid[:, t, :] = np.loadtxt(dir_seds + '/' + sed_name +'_fluxredshiftmod.txt')

    numZbins = redshiftDistGrid.size - 1
    numZ = redshiftGrid.size

    numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
    numObjectsTarget = np.sum(1 for line in open(params['target_catFile']))
    redshiftsInTarget = ('redshift' in params['target_bandOrder'])
    Ncompress = params['Ncompress']

    firstLine = int(threadNum * numObjectsTarget / float(numThreads))
    lastLine = int(min(numObjectsTarget,(threadNum + 1) * numObjectsTarget / float(numThreads)))
    numLines = lastLine - firstLine

    if threadNum == 0:
        msg= 'Number of Training Objects ' +  str(numObjectsTraining)
        logger.info(msg)

        msg='Number of Target Objects ' + str(numObjectsTarget)
        logger.info(msg)

    comm.Barrier()

    msg= 'Thread '+ str(threadNum) + ' , analyzes lines ' +  str(firstLine) + ' to ' + str( lastLine)
    logger.info(msg)

    DL = approx_DL()
    gp = PhotozGP(f_mod_interp,
              bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
              params['lines_pos'], params['lines_width'],
              params['V_C'], params['V_L'],
              params['alpha_C'], params['alpha_L'],
              redshiftGridGP, use_interpolators=True)

    # Create local files to store results
    numMetrics = 7 + len(params['confidenceLevels'])
    localPDFs = np.zeros((numLines, numZ))
    localMetrics = np.zeros((numLines, numMetrics))
    localCompressIndices = np.zeros((numLines,  Ncompress), dtype=int)
    localCompEvidences = np.zeros((numLines,  Ncompress))

    # Looping over chunks of the training set to prepare model predictions over z
    numChunks = params['training_numChunks']
    for chunk in range(numChunks):
        TR_firstLine = int(chunk * numObjectsTraining / float(numChunks))
        TR_lastLine = int(min(numObjectsTraining, (chunk + 1) * numObjectsTarget / float(numChunks)))
        targetIndices = np.arange(TR_firstLine, TR_lastLine)
        numTObjCk = TR_lastLine - TR_firstLine
        redshifts = np.zeros((numTObjCk, ))
        model_mean = np.zeros((numZ, numTObjCk, numBands))
        model_covar = np.zeros((numZ, numTObjCk, numBands))
        bestTypes = np.zeros((numTObjCk, ), dtype=int)
        ells = np.zeros((numTObjCk, ), dtype=int)

        # loop on training data and training GP coefficients produced by delight_learn
        # It fills the model_mean and model_covar predicted by GP
        loc = TR_firstLine - 1
        trainingDataIter = getDataFromFile(params, TR_firstLine, TR_lastLine,prefix="training_", ftype="gpparams")

        # loop on training data to load the GP parameter
        for loc, (z, ell, bands, X, B, flatarray) in enumerate(trainingDataIter):
            t1 = time()
            redshifts[loc] = z              # redshift of all training samples
            gp.setCore(X, B, nt,flatarray[0:nt+B+B*(B+1)//2])
            bestTypes[loc] = gp.bestType   # retrieve the best-type found by delight-learn
            ells[loc] = ell                # retrieve the luminosity parameter l

            # here is the model prediction of Gaussian Process for that particular trainning galaxy
            model_mean[:, loc, :], model_covar[:, loc, :] = gp.predictAndInterpolate(redshiftGrid, ell=ell)
            t2 = time()
            # print(loc, t2-t1)

        #Redshift prior on training galaxy
        # p_t = params['p_t'][bestTypes][None, :]
        # p_z_t = params['p_z_t'][bestTypes][None, :]
        # compute the prior for taht training sample
        prior = np.exp(-0.5*((redshiftGrid[:, None]-redshifts[None, :]) /params['zPriorSigma'])**2)
        # prior[prior < 1e-6] = 0
        # prior *= p_t * redshiftGrid[:, None] *
        # np.exp(-0.5 * redshiftGrid[:, None]**2 / p_z_t) / p_z_t

        if params['useCompression'] and params['compressionFilesFound']:
            fC = open(params['compressMargLikFile'])
            fCI = open(params['compressIndicesFile'])
            itCompM = itertools.islice(fC, firstLine, lastLine)
            iterCompI = itertools.islice(fCI, firstLine, lastLine)

        targetDataIter = getDataFromFile(params, firstLine, lastLine,prefix="target_", getXY=False, CV=False)

        # loop on target samples
        for loc, (z, normedRefFlux, bands, fluxes, fluxesVar, bCV, dCV, dVCV) in enumerate(targetDataIter):
            t1 = time()
            ell_hat_z = normedRefFlux * 4 * np.pi * params['fluxLuminosityNorm'] * (DL(redshiftGrid)**2. * (1+redshiftGrid))
            ell_hat_z[:] = 1
            if params['useCompression'] and params['compressionFilesFound']:
                indices = np.array(next(iterCompI).split(' '), dtype=int)
                sel = np.in1d(targetIndices, indices, assume_unique=True)
                # same likelihood as for template fitting
                like_grid2 = approx_flux_likelihood(fluxes,fluxesVar,model_mean[:, sel, :][:, :, bands],
                f_mod_covar=model_covar[:, sel, :][:, :, bands],
                marginalizeEll=True, normalized=False,
                ell_hat=ell_hat_z,
                ell_var=(ell_hat_z*params['ellPriorSigma'])**2)
                like_grid *= prior[:, sel]
            else:
                like_grid = np.zeros((nz, model_mean.shape[1]))
                # same likelihood as for template fitting, but cython
                approx_flux_likelihood_cy(
                    like_grid, nz, model_mean.shape[1], bands.size,
                    fluxes, fluxesVar,  # target galaxy fluxes and variance
                    model_mean[:, :, bands],     # prediction with Gaussian process
                    model_covar[:, :, bands],
                    ell_hat=ell_hat_z,           # it will find internally the ell
                    ell_var=(ell_hat_z*params['ellPriorSigma'])**2)
                like_grid *= prior[:, :] #likelihood multiplied by redshift training galaxies priors
            t2 = time()
            localPDFs[loc, :] += like_grid.sum(axis=1)  # the final redshift posterior is sum over training galaxies posteriors

            # compute the evidence for each model
            evidences = np.trapz(like_grid, x=redshiftGrid, axis=0)
            t3 = time()

            if params['useCompression'] and not params['compressionFilesFound']:
                if localCompressIndices[loc, :].sum() == 0:
                    sortind = np.argsort(evidences)[::-1][0:Ncompress]
                    localCompressIndices[loc, :] = targetIndices[sortind]
                    localCompEvidences[loc, :] = evidences[sortind]
                else:
                    dind = np.concatenate((targetIndices,localCompressIndices[loc, :]))
                    devi = np.concatenate((evidences,localCompEvidences[loc, :]))
                    sortind = np.argsort(devi)[::-1][0:Ncompress]
                    localCompressIndices[loc, :] = dind[sortind]
                    localCompEvidences[loc, :] = devi[sortind]

            if chunk == numChunks - 1\
                    and redshiftsInTarget\
                 and localPDFs[loc, :].sum() > 0:
                localMetrics[loc, :] = computeMetrics(z, redshiftGrid,localPDFs[loc, :],params['confidenceLevels'])
            t4 = time()
            if loc % 100 == 0:
                print(loc, t2-t1, t3-t2, t4-t3)

        if params['useCompression'] and params['compressionFilesFound']:
            fC.close()
            fCI.close()

    comm.Barrier()

    if threadNum == 0:
        globalPDFs = np.zeros((numObjectsTarget, numZ))
        globalCompressIndices = np.zeros((numObjectsTarget, Ncompress), dtype=int)
        globalCompEvidences = np.zeros((numObjectsTarget, Ncompress))
        globalMetrics = np.zeros((numObjectsTarget, numMetrics))
    else:
        globalPDFs = None
        globalCompressIndices = None
        globalCompEvidences = None
        globalMetrics = None

    firstLines = [int(k*numObjectsTarget/numThreads) for k in range(numThreads)]
    lastLines = [int(min(numObjectsTarget, (k+1)*numObjectsTarget/numThreads)) for k in range(numThreads)]
    numLines = [lastLines[k] - firstLines[k] for k in range(numThreads)]

    sendcounts = tuple([numLines[k] * numZ for k in range(numThreads)])
    displacements = tuple([firstLines[k] * numZ for k in range(numThreads)])
    comm.Gatherv(localPDFs,[globalPDFs, sendcounts, displacements, MPI.DOUBLE])

    sendcounts = tuple([numLines[k] * Ncompress for k in range(numThreads)])
    displacements = tuple([firstLines[k] * Ncompress for k in range(numThreads)])
    comm.Gatherv(localCompressIndices,[globalCompressIndices, sendcounts, displacements, MPI.LONG])
    comm.Gatherv(localCompEvidences,[globalCompEvidences, sendcounts, displacements, MPI.DOUBLE])
    comm.Barrier()

    sendcounts = tuple([numLines[k] * numMetrics for k in range(numThreads)])
    displacements = tuple([firstLines[k] * numMetrics for k in range(numThreads)])
    comm.Gatherv(localMetrics,[globalMetrics, sendcounts, displacements, MPI.DOUBLE])

    comm.Barrier()

    if threadNum == 0:
        fmt = '%.2e'
        fname = params['redshiftpdfFileComp'] if params['compressionFilesFound']\
            else params['redshiftpdfFile']
        np.savetxt(fname, globalPDFs, fmt=fmt)
        if redshiftsInTarget:
            np.savetxt(params['metricsFile'], globalMetrics, fmt=fmt)
        if params['useCompression'] and not params['compressionFilesFound']:
            np.savetxt(params['compressMargLikFile'],globalCompEvidences, fmt=fmt)
            np.savetxt(params['compressIndicesFile'],globalCompressIndices, fmt="%i")


#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # execute only if run as a script


    msg="Start Delight Learn.py"
    logger.info(msg)
    logger.info("--- Process Delight Learn ---")


    if len(sys.argv) < 2:
        raise Exception('Please provide a parameter file')

    delightApply(sys.argv[1])
