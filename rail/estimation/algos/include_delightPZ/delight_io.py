# -*- coding: utf-8 -*-

import numpy as np
import os
import collections
import configparser
import itertools
from delight.utils import approx_DL
from scipy.interpolate import interp1d


def parseParamFile(fileName, verbose=True, catFilesNeeded=True):
    """
    Parser for configuration inputtype parameter files,
    see examples for details. A bunch of them ar parsed.
    """
    config = configparser.ConfigParser()
    if not os.path.isfile(fileName):
        raise Exception(fileName+' : file not found')
    config.read(fileName)
    config.sections()

    for secName in ['Bands', 'Training', 'Target', 'Other']:
        if not config.has_section(secName):
            raise Exception(secName+' not found in parameter file')

    params = collections.OrderedDict()

    params['rootDir'] = config.get('Other', 'rootDir')
    if not os.path.isdir(params['rootDir']):
        raise Exception(params['rootDir']+' is not a valid directory')

    # Parsing Bands
    params['bands_directory'] = config.get('Bands', 'directory')
    if not os.path.isdir(params['bands_directory']):
        raise Exception(params['bands_directory']+' is not a valid directory')
    params['bandNames'] = config.get('Bands', 'Names').split(' ')

    # Parsing Templates
    params['templates_directory'] = config.get('Templates', 'directory')
    params['lambdaRef'] = config.getfloat('Templates', 'lambdaRef')
    params['templates_names'] = config.get('Templates', 'names').split(' ')
    params['p_t']\
        = np.array([float(x) for x in
                    config.get('Templates', 'p_t').split(' ')])
    params['p_z_t']\
        = np.array([float(x) for x in
                    config.get('Templates', 'p_z_t').split(' ')])
    assert params['p_z_t'].size == params['p_z_t'].size and\
        params['p_z_t'].size == len(params['templates_names'])

    # Parsing Training
    params['training_numChunks'] = config.getint('Training', 'numChunks')
    params['training_paramFile'] = config.get('Training', 'paramFile')
    params['training_catFile'] = config.get('Training', 'catFile')
    if catFilesNeeded and not os.path.isfile(params['training_catFile']):
        raise Exception(params['training_catFile']+' : file does not exist')
    params['training_referenceBand'] = config.get('Training', 'referenceBand')
    if params['training_referenceBand'] not in params['bandNames']:
        raise Exception(params['training_referenceBand']+' : is not a valid')
    params['training_bandOrder']\
        = config.get('Training', 'bandOrder').split(' ')
    params['training_extraFracFluxError']\
        = config.getfloat('Training', 'extraFracFluxError')
    for band in params['training_bandOrder']:
        if (band not in params['bandNames'])\
                and (band[:-4] not in params['bandNames'])\
                and (band != '_')\
                and (band != 'redshift'):
            raise Exception(band+' does not exist')
    if 'redshift' not in params['training_bandOrder']:
        raise Exception('redshift should be included in training')
    params['training_crossValidate'] =\
        config.getboolean('Training', 'crossValidate')
    params['training_CV_bandOrder']\
        = config.get('Training', 'crossValidationBandOrder').split(' ')
    params['training_CVfile'] = config.get('Training', 'CVfile')
    for band in params['training_CV_bandOrder']:
        if (band not in params['bandNames'])\
                and (band[:-4] not in params['bandNames'])\
                and (band != '_')\
                and (band != 'redshift'):
            raise Exception(band+' does not exist')

    # Simulation
    params['trainingFile'] = config.get('Simulation', 'trainingFile')
    params['targetFile'] = config.get('Simulation', 'targetFile')
    params['numObjects'] = int(config.getfloat('Simulation', 'numObjects'))
    params['noiseLevel'] = config.getfloat('Simulation', 'noiseLevel')

    # Parsing Target
    params['target_extraFracFluxError']\
        = config.getfloat('Target', 'extraFracFluxError')
    params['target_catFile'] = config.get('Target', 'catFile')
    if catFilesNeeded and not os.path.isfile(params['target_catFile']):
        raise Exception(params['target_catFile']+' : file does not exist')
    params['target_bandOrder']\
        = config.get('Target', 'bandOrder').split(' ')
    params['target_referenceBand'] = config.get('Target', 'referenceBand')
    if params['target_referenceBand'] not in params['bandNames']:
        raise Exception(params['target_referenceBand']+' : is not a valid')
    for band in params['target_bandOrder']:
        if (band not in params['bandNames'])\
                and (band[:-4] not in params['bandNames'])\
                and (band != '_')\
                and (band != 'redshift'):
            raise Exception(band+' does not exist')
    params['compressIndicesFile'] = config.get('Target', 'compressIndicesFile')
    params['compressMargLikFile'] = config.get('Target', 'compressMargLikFile')
    if os.path.isfile(params['compressIndicesFile'])\
       and os.path.isfile(params['compressMargLikFile']):
            params['compressionFilesFound'] = True
    else:
        params['compressionFilesFound'] = False
    params['Ncompress'] = config.getint('Target', 'Ncompress')
    params['useCompression'] = config.getboolean("Target", 'useCompression')
    params['redshiftpdfFile'] = config.get('Target', 'redshiftpdfFile')
    params['redshiftpdfFileComp'] = config.get('Target', 'redshiftpdfFileComp')
    params['redshiftpdfFileTemp'] = config.get('Target', 'redshiftpdfFileTemp')
    params['metricsFile'] = config.get('Target', 'metricsFile')
    params['metricsFileTemp'] = config.get('Target', 'metricsFileTemp')

    # Parsing other parameters
    params['zPriorSigma'] = config.getfloat('Other', 'zPriorSigma')
    params['ellPriorSigma'] = config.getfloat('Other', 'ellPriorSigma')
    params['fluxLuminosityNorm']\
        = config.getfloat('Other', 'fluxLuminosityNorm')
    params['alpha_C'] = config.getfloat('Other', 'alpha_C')
    params['alpha_L'] = config.getfloat('Other', 'alpha_L')
    params['V_C'] = config.getfloat('Other', 'V_C')
    params['V_L'] = config.getfloat('Other', 'V_L')
    params['redshiftMin'] = config.getfloat('Other', 'redshiftMin')
    params['redshiftMax'] = config.getfloat('Other', 'redshiftMax')
    params['redshiftBinSize']\
        = config.getfloat('Other', 'redshiftBinSize')
    params['redshiftNumBinsGPpred']\
        = config.getint('Other', 'redshiftNumBinsGPpred')
    params['redshiftDisBinSize']\
        = config.getfloat('Other', 'redshiftDisBinSize')
    params['lines_pos']\
        = [float(x) for x in
           config.get('Other', 'lines_pos').split(' ')]
    params['lines_width']\
        = [float(x) for x in
           config.get('Other', 'lines_width').split(' ')]
    params['confidenceLevels']\
        = [float(x) for x in
           config.get('Other', 'confidenceLevels').split(' ')]

    if verbose:
        print('Input parameter file:', fileName)
        print('Parameters read:')
        for k, v in params.items():
            if type(v) is list:
                print('> ', "%-20s" % k, ' '.join([str(x) for x in v]))
            else:
                print('> ', "%-20s" % k, v)

    return params


def readColumnPositions(params, prefix="training_", refFlux=True):
    """
    Read column/band information needed for parsing catalog file,
        in particular the column positions.
    """
    bandIndices = np.array([ib for ib, b in enumerate(params['bandNames'])
                            if b in params[prefix+'bandOrder']])
    bandNames = np.array(params['bandNames'])[bandIndices]
    bandColumns = np.array([params[prefix+'bandOrder'].index(b)
                            for b in bandNames])
    bandVarColumns = np.array([params[prefix+'bandOrder'].index(b+'_var')
                               for b in bandNames])
    if 'redshift' in params[prefix+'bandOrder']:
        redshiftColumn = params[prefix+'bandOrder'].index('redshift')
    else:
        redshiftColumn = -1
    if refFlux:
        refBandColumn = params[prefix+'bandOrder']\
            .index(params[prefix+'referenceBand'])
        return bandIndices, bandNames, bandColumns, bandVarColumns,\
            redshiftColumn, refBandColumn
    else:
        return bandIndices, bandNames, bandColumns, bandVarColumns,\
            redshiftColumn


def readBandCoefficients(params):
    """
    Read band/filter information, in particular the Gaussian Mixture coefs.
    """
    bandCoefAmplitudes = []
    bandCoefPositions = []
    bandCoefWidths = []
    for band in params['bandNames']:
        fname = params['bands_directory'] + '/' + band\
            + '_gaussian_coefficients.txt'
        data = np.loadtxt(fname)
        bandCoefAmplitudes.append(data[:, 0])
        bandCoefPositions.append(data[:, 1])
        bandCoefWidths.append(data[:, 2])
    bandCoefAmplitudes = np.vstack(bandCoefAmplitudes)
    bandCoefPositions = np.vstack(bandCoefPositions)
    bandCoefWidths = np.vstack(bandCoefWidths)
    norms =\
        np.sqrt(2*np.pi) * np.sum(bandCoefAmplitudes * bandCoefWidths, axis=1)
    return bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms


def createGrids(params):
    """
    Create redshift grids from parameters in file.
    """
    redshiftDistGrid = np.arange(0, params['redshiftMax'],
                                 params['redshiftDisBinSize'])
    if True:
        redshiftGrid = np.arange(params['redshiftMin'],
                                 params['redshiftMax'],
                                 params['redshiftBinSize'])
    else:
        num = int((params['redshiftMax'] - params['redshiftMin']) /
                  params['redshiftBinSize'])
        redshiftGrid = np.logspace(np.log10(params['redshiftMin']),
                                   np.log10(params['redshiftMax']*1.01),
                                   num)
    redshiftGridGP = np.logspace(np.log10(params['redshiftMin']),
                                 np.log10(params['redshiftMax']*1.01),
                                 params['redshiftNumBinsGPpred'])
    return redshiftDistGrid, redshiftGrid, redshiftGridGP


def readSEDs(params):
    """
    Read SED parameters.
    """
    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
    f_mod = np.zeros((len(params['templates_names']),
                      len(params['bandNames'])), dtype=object)
    for it, sed_name in enumerate(params['templates_names']):
        data = np.loadtxt(params['templates_directory'] +
                          '/' + sed_name + '_fluxredshiftmod.txt')
        for jf in range(len(params['bandNames'])):
            f_mod[it, jf] = interp1d(redshiftGrid, data[:, jf],
                                     kind='linear', bounds_error=False,
                                     fill_value='extrapolate')
    return f_mod


def getDataFromFile(params, firstLine, lastLine,
                    prefix="", ftype="catalog", getXY=True, CV=False):
    """
    Returns an iterator to parse an input catalog file.
    Returns the fluxes, redshifts, etc, and also GP inputs if getXY=True.
    """

    if ftype == "gpparams":

        with open(params[prefix+'paramFile']) as f:
            for line in itertools.islice(f, firstLine, lastLine):
                data = np.fromstring(line, dtype=float, sep=' ')
                B = int(data[0])
                z = data[1]
                ell = data[2]
                bands = data[3:3+B]
                flatarray = data[3+B:]
                X = np.zeros((B, 3))
                for off, iband in enumerate(bands):
                    X[off, 0] = iband
                    X[off, 1] = z
                    X[off, 2] = ell

                yield z, ell, bands, X, B, flatarray

    if ftype == "catalog":

        DL = approx_DL()
        bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,\
            refBandColumn = readColumnPositions(params, prefix=prefix)
        bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
            = readBandCoefficients(params)
        refBandNorm = norms[params['bandNames']
                            .index(params[prefix+'referenceBand'])]

        if CV:
            bandIndicesCV, bandNamesCV, bandColumnsCV,\
                bandVarColumnsCV, redshiftColumnCV =\
                readColumnPositions(params, prefix=prefix+'CV_', refFlux=False)

        with open(params[prefix+'catFile']) as f:
            for line in itertools.islice(f, firstLine, lastLine):

                data = np.array(line.split(' '), dtype=float)
                refFlux = data[refBandColumn]
                normedRefFlux = refFlux * refBandNorm
                if redshiftColumn >= 0:
                    z = data[redshiftColumn]
                else:
                    z = -1

                # drop bad values and find how many bands are valid
                mask = np.isfinite(data[bandColumns])
                mask &= np.isfinite(data[bandVarColumns])
                mask &= data[bandColumns] > 0.0
                mask &= data[bandVarColumns] > 0.0
                bandsUsed = np.where(mask)[0]
                numBandsUsed = mask.sum()

                if z > -1:
                    ell = normedRefFlux * 4 * np.pi \
                        * params['fluxLuminosityNorm'] * DL(z)**2 * (1+z)

                if (refFlux <= 0) or (not np.isfinite(refFlux))\
                        or (z < 0) or (numBandsUsed <= 1):
                    print("Skipping galaxy: refflux=", refFlux,
                          "z=", z, "numBandsUsed=", numBandsUsed)
                    continue  # not valid data - skip to next valid object

                fluxes = data[bandColumns[mask]]
                fluxesVar = data[bandVarColumns[mask]] +\
                    (params['training_extraFracFluxError'] * fluxes)**2

                if CV:
                    maskCV = np.isfinite(data[bandColumnsCV])
                    maskCV &= np.isfinite(data[bandVarColumnsCV])
                    maskCV &= data[bandColumnsCV] > 0.0
                    maskCV &= data[bandVarColumnsCV] > 0.0
                    bandsUsedCV = np.where(maskCV)[0]
                    numBandsUsedCV = maskCV.sum()
                    fluxesCV = data[bandColumnsCV[maskCV]]
                    fluxesCVVar = data[bandVarColumnsCV[maskCV]] +\
                        (params['training_extraFracFluxError'] * fluxesCV)**2

                if not getXY:

                    if CV:
                        yield z, normedRefFlux,\
                            bandIndices[mask], fluxes, fluxesVar,\
                            bandIndicesCV[maskCV], fluxesCV, fluxesCVVar
                    else:
                        yield z, normedRefFlux,\
                            bandIndices[mask], fluxes, fluxesVar,\
                            None, None, None

                if getXY:

                    Y = np.zeros((numBandsUsed, 1))
                    Yvar = np.zeros((numBandsUsed, 1))
                    X = np.ones((numBandsUsed, 3))
                    for off, iband in enumerate(bandIndices[mask]):
                        X[off, 0] = iband
                        X[off, 1] = z
                        X[off, 2] = ell
                        Y[off, 0] = fluxes[off]
                        Yvar[off, 0] = fluxesVar[off]

                    if CV:
                        yield z, normedRefFlux,\
                            bandIndices[mask], fluxes, fluxesVar,\
                            bandIndicesCV[maskCV], fluxesCV, fluxesCVVar,\
                            X, Y, Yvar
                    else:
                        yield z, normedRefFlux,\
                            bandIndices[mask], fluxes, fluxesVar,\
                            None, None, None,\
                            X, Y, Yvar
