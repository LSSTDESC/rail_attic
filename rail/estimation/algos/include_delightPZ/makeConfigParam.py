####################################################################################################
# Script name : makeConfigParam.py
#
#  Generate Config parameter required by Delight
#
#  Some parameters are read from the from the rail configuration file
#  Some other parameter are hardcoded in this file
# The fina goal is to retrieve those parameters from RAIL config file
#####################################################################################################
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import leastsq

from delight.utils import *
from delight.io import *

import coloredlogs
import logging


import os
import yaml

from pkg_resources import resource_filename

# Create a logger object.
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s %(name)s[%(process)d] %(levelname)s %(message)s')


def makeConfigParam(path,inputs_rail):
    """
    makeConfigParam(path,inputs_rail)

    generate Configuration parameter file in ascii. This file is decoded by Delight functions with argparse

    : inputs:
      -  path : where the FILTERS and SEDs datafiles used by Delight initialisation are stored,
      - inputs_rail : RAIL parameter files

    Either the parameters used by Delight are hardcoded here of the can be setup by RAIL config strcture (yaml) in inputs_rail

    :return: paramfile_txt , the string for the configuration file. RAIL will write itself this file.
    """

    logger.debug("__name__:"+__name__)
    logger.debug("__file__"+__file__)

    msg = "----- makeConfigParam ------"
    logger.info(msg)

    logger.debug(" received path = "+ path)
    #logger.debug(" received input_rail = " + inputs_rail)

    # 1) Let 's create a parameter file from scratch.

    #paramfile_txt = "\n"
    #paramfile_txt += \
    paramfile_txt = \
"""
# DELIGHT parameter file
# Syntactic rules:
# - You can set parameters with : or =
# - Lines starting with # or ; will be ignored
# - Multiple values (band names, band orders, confidence levels)
#   must beb separated by spaces
# - The input files should contain numbers separated with spaces.
# - underscores mean unused column
"""

    # 2) Filter Section
    paramfile_txt += "\n"
    paramfile_txt += \
"""
[Bands]
names: lsst_u lsst_g lsst_r lsst_i lsst_z lsst_y
"""

    paramfile_txt += "directory: " + os.path.join(path, 'FILTERS')

    paramfile_txt +=  \
"""
numCoefs: 15
bands_verbose: True
bands_debug: True
bands_makeplots: False
"""

    # 3) Template Section

    paramfile_txt +=  \
"""
[Templates]
"""
    paramfile_txt += "directory: " + os.path.join(path, 'CWW_SEDs')

    paramfile_txt +=  \
"""
names: El_B2004a Sbc_B2004a Scd_B2004a SB3_B2004a SB2_B2004a Im_B2004a ssp_25Myr_z008 ssp_5Myr_z008
p_t: 0.27 0.26 0.25 0.069 0.021 0.11 0.0061 0.0079
p_z_t:0.23 0.39 0.33 0.31 1.1 0.34 1.2 0.14
lambdaRef: 4.5e3
"""

    # 4) Simulation Section

    paramfile_txt +=  \
"""
[Simulation]
numObjects: 1000
noiseLevel: 0.03
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
trainingFile: data_lsst/galaxies-fluxredshifts.txt
targetFile: data_lsst/galaxies-fluxredshifts2.txt
"""
    else:
        thepath=inputs_rail["tempdatadir"]
        paramfile_txt += "trainingFile: " + os.path.join(thepath, 'galaxies-fluxredshifts.txt')
        paramfile_txt += "\n"
        paramfile_txt += "targetFile: " +  os.path.join(thepath, 'galaxies-fluxredshifts2.txt')
        paramfile_txt += "\n"



    # 5) Training Section

    paramfile_txt +=  \
"""
[Training]
"""
    if inputs_rail == None:
        paramfile_txt +=  \
"""
catFile: data_lsst/galaxies-fluxredshifts.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        paramfile_txt += "catFile: " + os.path.join(thepath, 'galaxies-fluxredshifts.txt')

    paramfile_txt +=  \
"""
bandOrder: lsst_u lsst_u_var lsst_g lsst_g_var lsst_r lsst_r_var lsst_i lsst_i_var lsst_z lsst_z_var lsst_y lsst_y_var redshift
referenceBand: lsst_i
extraFracFluxError: 1e-4
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
paramFile: data_lsst/galaxies-gpparams.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        paramfile_txt += "paramFile: " + os.path.join(thepath, 'galaxies-gpparams.txt')


    paramfile_txt +=  \
"""
crossValidate: False
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
CVfile: data_lsst/galaxies-gpCV.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        paramfile_txt += "CVfile: " + os.path.join(thepath, 'galaxies-gpCV.txt')

    paramfile_txt +=  \
"""
crossValidationBandOrder: _ _ _ _ lsst_r lsst_r_var _ _ _ _ _ _
numChunks: 1
"""

    # 6) Estimation Section


    paramfile_txt +=  \
"""
[Target]
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
catFile: data_lsst/galaxies-fluxredshifts2.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        paramfile_txt += "catFile: " + os.path.join(thepath, 'galaxies-fluxredshifts2.txt')

    paramfile_txt +=  \
"""
bandOrder: lsst_u lsst_u_var lsst_g lsst_g_var lsst_r lsst_r_var lsst_i lsst_i_var lsst_z lsst_z_var lsst_y lsst_y_var redshift
referenceBand: lsst_r
extraFracFluxError: 1e-4
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
redshiftpdfFile: data_lsst/galaxies-redshiftpdfs.txt
redshiftpdfFileTemp: data_lsst/galaxies-redshiftpdfs-cww.txt
metricsFile:  data_lsst/galaxies-redshiftmetrics.txt
metricsFileTemp:  data_lsst/galaxies-redshiftmetrics-cww.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        paramfile_txt += "redshiftpdfFile: " + os.path.join(thepath, 'galaxies-redshiftpdfs.txt')
        paramfile_txt += "\n"
        paramfile_txt += "redshiftpdfFileTemp: " + os.path.join(thepath, 'galaxies-redshiftpdfs-cww.txt')
        paramfile_txt += "\n"
        paramfile_txt += "metricsFile: " + os.path.join(thepath, 'galaxies-redshiftmetrics.txt')
        paramfile_txt += "\n"
        paramfile_txt += "metricsFileTemp: " + os.path.join(thepath, 'galaxies-redshiftmetrics-cww.txt')


    paramfile_txt +=  \
"""
useCompression: False
Ncompress: 10
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
compressIndicesFile: data_lsst/galaxies-compressionIndices.txt
compressMargLikFile: data_lsst/galaxies-compressionMargLikes.txt
redshiftpdfFileComp: data_lsst/galaxies-redshiftpdfs-comp.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        paramfile_txt += "compressIndicesFile: " + os.path.join(thepath, 'galaxies-compressionIndices.txt')
        paramfile_txt += "\n"
        paramfile_txt += "compressMargLikFile: " + os.path.join(thepath, 'galaxies-compressionMargLikes.txt')
        paramfile_txt += "\n"
        paramfile_txt += "redshiftpdfFileComp: " + os.path.join(thepath, 'galaxies-redshiftpdfs-comp.txt')
        paramfile_txt += "\n"

    # 7) Other Section

    if inputs_rail == None:
        paramfile_txt +=  \
"""
[Other]
rootDir: ./
zPriorSigma: 0.2
ellPriorSigma: 0.5
fluxLuminosityNorm: 1.0
alpha_C: 1.0e3
V_C: 0.1
alpha_L: 1.0e2
V_L: 0.1
lines_pos: 6500 5002.26 3732.22
lines_width: 20.0 20.0 20.0
"""
    else:
        zPriorSigma = inputs_rail["zPriorSigma"]
        ellPriorSigma = inputs_rail["ellPriorSigma"]
        fluxLuminosityNorm = inputs_rail["fluxLuminosityNorm"]
        alpha_C = inputs_rail["alpha_C"]
        V_C = inputs_rail["V_C"]
        alpha_L = inputs_rail["alpha_L"]
        V_L = inputs_rail["V_L"]
        lineWidthSigma = inputs_rail["lineWidthSigma"]

        paramfile_txt += \
"""
[Other]
rootDir: ./
"""

        paramfile_txt += "zPriorSigma: " + str(zPriorSigma)
        paramfile_txt += "\n"
        paramfile_txt += "ellPriorSigma: " + str(ellPriorSigma)
        paramfile_txt += "\n"
        paramfile_txt += "fluxLuminosityNorm: " + str(fluxLuminosityNorm)
        paramfile_txt += "\n"
        paramfile_txt += "alpha_C: " + str(alpha_C)
        paramfile_txt += "\n"
        paramfile_txt += "V_C: " + str(V_C)
        paramfile_txt += "\n"
        paramfile_txt += "alpha_L: " + str(alpha_L)
        paramfile_txt += "\n"
        paramfile_txt += "V_L: " + str(V_L)
        paramfile_txt += "\n"
        paramfile_txt += "lines_pos: 6500 5002.26 3732.22 \n"
        paramfile_txt += "\n"
        paramfile_txt += "lines_width: " + str(lineWidthSigma) + " " + \
                         str(lineWidthSigma) + " " + \
                         str(lineWidthSigma) + " " + \
                         str(lineWidthSigma) + " " + "\n"


    if inputs_rail == None:
        paramfile_txt +=  \
"""
redshiftMin: 0.1
redshiftMax: 1.101
redshiftNumBinsGPpred: 100
redshiftBinSize: 0.001
redshiftDisBinSize: 0.2
"""
    else:

        msg = "Decode redshift parameter from RAIL config file"
        logger.debug(msg)

        dlght_redshiftMin           = inputs_rail["dlght_redshiftMin"]
        dlght_redshiftMax           = inputs_rail["dlght_redshiftMax"]
        dlght_redshiftNumBinsGPpred = inputs_rail["dlght_redshiftNumBinsGPpred"]
        dlght_redshiftBinSize       = inputs_rail["dlght_redshiftBinSize"]
        dlght_redshiftDisBinSize    = inputs_rail["dlght_redshiftDisBinSize"]

        # will check later what to do with these parameters

        paramfile_txt += "redshiftMin: " + str(dlght_redshiftMin)
        paramfile_txt += "\n"
        paramfile_txt += "redshiftMax: " + str(dlght_redshiftMax)
        paramfile_txt += "\n"
        paramfile_txt += "redshiftNumBinsGPpred: " + str(dlght_redshiftNumBinsGPpred)
        paramfile_txt += "\n"
        paramfile_txt += "redshiftBinSize: " + str(dlght_redshiftBinSize)
        paramfile_txt += "\n"
        paramfile_txt += "redshiftDisBinSize: " + str(dlght_redshiftDisBinSize)
        paramfile_txt += "\n"




    paramfile_txt += \
"""
confidenceLevels: 0.1 0.50 0.68 0.95
"""


    return paramfile_txt






def makeConfigParamChunk(path,inputs_rail,chunknum):
    """
    makeConfigParam(path,inputs_rail)

    generate Configuration parameter file in ascii. This file is decoded by Delight functions with argparse

    : inputs:
      -  path : where the FILTERS and SEDs datafiles used by Delight initialisation are stored,
      - inputs_rail : RAIL parameter files
      - chunknum : chunknumber

    Either the parameters used by Delight are hardcoded here of the can be setup by RAIL config strcture (yaml) in inputs_rail

    :return: paramfile_txt , the string for the configuration file. RAIL will write itself this file.
    """

    logger.debug("__name__:"+__name__)
    logger.debug("__file__"+__file__)

    msg = "----- makeConfigParamChunk ------"
    logger.info(msg)

    logger.debug(" received path = "+ path)
    #logger.debug(" received input_rail = " + inputs_rail)

    # 1) Let 's create a parameter file from scratch.

    #paramfile_txt = "\n"
    #paramfile_txt += \
    paramfile_txt = \
"""
# DELIGHT parameter file
# Syntactic rules:
# - You can set parameters with : or =
# - Lines starting with # or ; will be ignored
# - Multiple values (band names, band orders, confidence levels)
#   must beb separated by spaces
# - The input files should contain numbers separated with spaces.
# - underscores mean unused column
"""

    # 2) Filter Section
    paramfile_txt += "\n"
    paramfile_txt += \
"""
[Bands]
names: lsst_u lsst_g lsst_r lsst_i lsst_z lsst_y
"""

    paramfile_txt += "directory: " + os.path.join(path, 'FILTERS')

    paramfile_txt +=  \
"""
numCoefs: 15
bands_verbose: True
bands_debug: True
bands_makeplots: False
"""

    # 3) Template Section

    paramfile_txt +=  \
"""
[Templates]
"""
    paramfile_txt += "directory: " + os.path.join(path, 'CWW_SEDs')

    paramfile_txt +=  \
"""
names: El_B2004a Sbc_B2004a Scd_B2004a SB3_B2004a SB2_B2004a Im_B2004a ssp_25Myr_z008 ssp_5Myr_z008
p_t: 0.27 0.26 0.25 0.069 0.021 0.11 0.0061 0.0079
p_z_t:0.23 0.39 0.33 0.31 1.1 0.34 1.2 0.14
lambdaRef: 4.5e3
"""

    # 4) Simulation Section

    paramfile_txt +=  \
"""
[Simulation]
numObjects: 1000
noiseLevel: 0.03
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
trainingFile: data_lsst/galaxies-fluxredshifts.txt
targetFile: data_lsst/galaxies-fluxredshifts2.txt
"""
    else:
        thepath=inputs_rail["tempdatadir"]
        paramfile_txt += "trainingFile: " + os.path.join(thepath, 'galaxies-fluxredshifts.txt')
        paramfile_txt += "\n"
        ####
        thepath = inputs_rail["tempdatadir"]
        filename = 'galaxies-fluxredshifts2_{}.txt'.format(chunknum)
        paramfile_txt += "targetFile: " +  os.path.join(thepath, filename)
        paramfile_txt += "\n"

    # 5) Training Section

    paramfile_txt +=  \
"""
[Training]
"""
    if inputs_rail == None:
        paramfile_txt +=  \
"""
catFile: data_lsst/galaxies-fluxredshifts.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        paramfile_txt += "catFile: " + os.path.join(thepath, 'galaxies-fluxredshifts.txt')

    paramfile_txt +=  \
"""
bandOrder: lsst_u lsst_u_var lsst_g lsst_g_var lsst_r lsst_r_var lsst_i lsst_i_var lsst_z lsst_z_var lsst_y lsst_y_var redshift
referenceBand: lsst_i
extraFracFluxError: 1e-4
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
paramFile: data_lsst/galaxies-gpparams.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        paramfile_txt += "paramFile: " + os.path.join(thepath, 'galaxies-gpparams.txt')


    paramfile_txt +=  \
"""
crossValidate: False
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
CVfile: data_lsst/galaxies-gpCV.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        paramfile_txt += "CVfile: " + os.path.join(thepath, 'galaxies-gpCV.txt')

    paramfile_txt +=  \
"""
crossValidationBandOrder: _ _ _ _ lsst_r lsst_r_var _ _ _ _ _ _
numChunks: 1
"""

    # 6) Estimation Section


    paramfile_txt +=  \
"""
[Target]
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
catFile: data_lsst/galaxies-fluxredshifts2.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        filename = 'galaxies-fluxredshifts2_{}.txt'.format(chunknum)
        paramfile_txt += "catFile: " + os.path.join(thepath, filename)

    paramfile_txt +=  \
"""
bandOrder: lsst_u lsst_u_var lsst_g lsst_g_var lsst_r lsst_r_var lsst_i lsst_i_var lsst_z lsst_z_var lsst_y lsst_y_var redshift
referenceBand: lsst_r
extraFracFluxError: 1e-4
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
redshiftpdfFile: data_lsst/galaxies-redshiftpdfs.txt
redshiftpdfFileTemp: data_lsst/galaxies-redshiftpdfs-cww.txt
metricsFile:  data_lsst/galaxies-redshiftmetrics.txt
metricsFileTemp:  data_lsst/galaxies-redshiftmetrics-cww.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        filename = 'galaxies-redshiftpdfs_{}.txt'.format(chunknum)
        paramfile_txt += "redshiftpdfFile: " + os.path.join(thepath, filename)
        paramfile_txt += "\n"
        filename = 'galaxies-redshiftpdfs-cww_{}.txt'.format(chunknum)
        paramfile_txt += "redshiftpdfFileTemp: " + os.path.join(thepath, filename)
        paramfile_txt += "\n"
        filename = 'galaxies-redshiftmetrics_{}.txt'.format(chunknum)
        paramfile_txt += "metricsFile: " + os.path.join(thepath, filename)
        paramfile_txt += "\n"
        filename = 'galaxies-redshiftmetrics-cww_{}.txt'.format(chunknum)
        paramfile_txt += "metricsFileTemp: " + os.path.join(thepath, filename)


    paramfile_txt +=  \
"""
useCompression: False
Ncompress: 10
"""

    if inputs_rail == None:
        paramfile_txt +=  \
"""
compressIndicesFile: data_lsst/galaxies-compressionIndices.txt
compressMargLikFile: data_lsst/galaxies-compressionMargLikes.txt
redshiftpdfFileComp: data_lsst/galaxies-redshiftpdfs-comp.txt
"""
    else:
        thepath = inputs_rail["tempdatadir"]
        filename = 'galaxies-compressionIndices_{}.txt'.format(chunknum)
        paramfile_txt += "compressIndicesFile: " + os.path.join(thepath, filename)
        paramfile_txt += "\n"
        filename = 'galaxies-compressionMargLikes_{}.txt'.format(chunknum)
        paramfile_txt += "compressMargLikFile: " + os.path.join(thepath, filename)
        paramfile_txt += "\n"
        filename = 'galaxies-redshiftpdfs-comp_{}.txt'.format(chunknum)
        paramfile_txt += "redshiftpdfFileComp: " + os.path.join(thepath, filename)
        paramfile_txt += "\n"

    # 7) Other Section

    if inputs_rail == None:
        paramfile_txt +=  \
"""
[Other]
rootDir: ./
zPriorSigma: 0.2
ellPriorSigma: 0.5
fluxLuminosityNorm: 1.0
alpha_C: 1.0e3
V_C: 0.1
alpha_L: 1.0e2
V_L: 0.1
lines_pos: 6500 5002.26 3732.22
lines_width: 20.0 20.0 20.0
"""
    else:
        zPriorSigma = inputs_rail["zPriorSigma"]
        ellPriorSigma = inputs_rail["ellPriorSigma"]
        fluxLuminosityNorm =  inputs_rail["fluxLuminosityNorm"]
        alpha_C  = inputs_rail["alpha_C"]
        V_C = inputs_rail["V_C"]
        alpha_L = inputs_rail["alpha_L"]
        V_L = inputs_rail["V_L"]
        lineWidthSigma = inputs_rail["lineWidthSigma"]

        paramfile_txt += \
"""
[Other]
rootDir: ./
"""

        paramfile_txt += "zPriorSigma: "+ str(zPriorSigma)
        paramfile_txt += "\n"
        paramfile_txt += "ellPriorSigma: " + str(ellPriorSigma)
        paramfile_txt += "\n"
        paramfile_txt += "fluxLuminosityNorm: " + str(fluxLuminosityNorm)
        paramfile_txt += "\n"
        paramfile_txt += "alpha_C: " + str(alpha_C)
        paramfile_txt += "\n"
        paramfile_txt += "V_C: " + str(V_C)
        paramfile_txt += "\n"
        paramfile_txt += "alpha_L: " + str(alpha_L)
        paramfile_txt += "\n"
        paramfile_txt += "V_L: " + str(V_L)
        paramfile_txt += "\n"
        paramfile_txt += "lines_pos: 6500 5002.26 3732.22 \n"
        paramfile_txt += "\n"
        paramfile_txt += "lines_width: " + str(lineWidthSigma) + " " +\
                         str(lineWidthSigma) + " " +\
                         str(lineWidthSigma) + " " + \
                         str(lineWidthSigma) + " " + "\n"






    if inputs_rail == None:
        paramfile_txt +=  \
"""
redshiftMin: 0.1
redshiftMax: 1.101
redshiftNumBinsGPpred: 100
redshiftBinSize: 0.001
redshiftDisBinSize: 0.2
"""
    else:

        msg = "Decode redshift parameter from RAI config file"
        logger.debug(msg)

        dlght_redshiftMin           = inputs_rail["dlght_redshiftMin"]
        dlght_redshiftMax           = inputs_rail["dlght_redshiftMax"]
        dlght_redshiftNumBinsGPpred = inputs_rail["dlght_redshiftNumBinsGPpred"]
        dlght_redshiftBinSize       = inputs_rail["dlght_redshiftBinSize"]
        dlght_redshiftDisBinSize    = inputs_rail["dlght_redshiftDisBinSize"]

        # will check later what to do with these parameters

        paramfile_txt += "redshiftMin: " + str(dlght_redshiftMin)
        paramfile_txt += "\n"
        paramfile_txt += "redshiftMax: " + str(dlght_redshiftMax)
        paramfile_txt += "\n"
        paramfile_txt += "redshiftNumBinsGPpred: " + str(dlght_redshiftNumBinsGPpred)
        paramfile_txt += "\n"
        paramfile_txt += "redshiftBinSize: " + str(dlght_redshiftBinSize)
        paramfile_txt += "\n"
        paramfile_txt += "redshiftDisBinSize: " + str(dlght_redshiftDisBinSize)
        paramfile_txt += "\n"




    paramfile_txt += \
"""
confidenceLevels: 0.1 0.50 0.68 0.95
"""


    return paramfile_txt

#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # execute only if run as a script


    msg="Start  makeConfigParam."
    logger.info(msg)
    logger.info("--- Make configuration parameter ---")

    logger.debug("__name__:"+__name__)
    logger.debug("__file__:"+__file__)

    datapath=resource_filename('delight', '../data')

    logger.debug("datapath = " + datapath)



    param_txt=makeConfigParam(datapath,None)

    logger.info(param_txt)
