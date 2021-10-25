#######################################################################################################
#
# script : convertDESCcat.py
#
# convert DESC catalog to be injected in Delight
# produce files `galaxies-redshiftpdfs.txt` and `galaxies-redshiftpdfs2.txt` for training and target
#
#########################################################################################################


import sys
import os
import numpy as np
from functools import reduce

import pprint

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from delight.io import *
from delight.utils import *

import coloredlogs
import logging

import h5py

#from interfaces.rail.utils  import load_training_data, get_input_data_size_hdf5,load_raw_hdf5_data
#from rail.estimation.utils  import load_training_data, get_input_data_size_hdf5,load_raw_hdf5_data
from rail.estimation.algos.include_delightPZ.utils import load_training_data, get_input_data_size_hdf5,load_raw_hdf5_data

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s, %(name)s[%(process)d] %(levelname)s %(message)s')

# option to convert DC2 flux level (in AB units) into internal Delight units
# this option will be removed when optimisation of parameters will be implemented
FLAG_CONVERTFLUX_TODELIGHTUNIT=True



def group_entries(f):
    """
    group entries in single numpy array

    """
    galid = f['id'][()][:, np.newaxis]
    redshift = f['redshift'][()][:, np.newaxis]
    mag_err_g_lsst = f['mag_err_g_lsst'][()][:, np.newaxis]
    mag_err_i_lsst = f['mag_err_i_lsst'][()][:, np.newaxis]
    mag_err_r_lsst = f['mag_err_r_lsst'][()][:, np.newaxis]
    mag_err_u_lsst = f['mag_err_u_lsst'][()][:, np.newaxis]
    mag_err_y_lsst = f['mag_err_y_lsst'][()][:, np.newaxis]
    mag_err_z_lsst = f['mag_err_z_lsst'][()][:, np.newaxis]
    mag_g_lsst = f['mag_g_lsst'][()][:, np.newaxis]
    mag_i_lsst = f['mag_i_lsst'][()][:, np.newaxis]
    mag_r_lsst = f['mag_r_lsst'][()][:, np.newaxis]
    mag_u_lsst = f['mag_u_lsst'][()][:, np.newaxis]
    mag_y_lsst = f['mag_y_lsst'][()][:, np.newaxis]
    mag_z_lsst = f['mag_z_lsst'][()][:, np.newaxis]

    full_arr = np.hstack((galid, redshift, mag_u_lsst, mag_g_lsst, mag_r_lsst, mag_i_lsst, mag_z_lsst, mag_y_lsst, \
                          mag_err_u_lsst, mag_err_g_lsst, mag_err_r_lsst, mag_err_i_lsst, mag_err_z_lsst,
                          mag_err_y_lsst))
    return full_arr


def filter_mag_entries(d,nb=6):
    """
    Filter bad data with bad magnitudes

    input
      - d: array of magnitudes and errors
      - nb : number of bands
    output :
      - indexes of row to be filtered

    """

    u = d[:, 2]
    idx_u = np.where(u > 31.8)[0]

    return idx_u


def mag_to_flux(d,nb=6):
    """

    Convert magnitudes to fluxes

    input:
       -d : array of magnitudes with errors


    :return:
        array of fluxes with error
    """

    fluxes = np.zeros_like(d)

    fluxes[:, 0] = d[:, 0]  # object index
    fluxes[:, 1] = d[:, 1]  # redshift

    for idx in np.arange(nb):
        fluxes[:, 2 + idx] = np.power(10, -0.4 * d[:, 2 + idx]) # fluxes
        fluxes[:, 8 + idx] = fluxes[:, 2 + idx] * d[:, 8 + idx] # errors on fluxes
    return fluxes



def filter_flux_entries(d,nb=6,nsig=5):
    """
    Filter noisy data on the the number SNR

    input :
     - d: flux and errors array
     - nb : number of bands
     - nsig : number of sigma

     output:
        indexes of row to suppress

    """


    # collection of indexes
    indexes = []
    #indexes = np.array(indexes, dtype=np.int)
    indexes = np.array(indexes, dtype=int)

    for idx in np.arange(nb):
        ratio = d[:, 2 + idx] / d[:, 8 + idx]  # flux divided by sigma-flux
        bad_indexes = np.where(ratio < nsig)[0]
        indexes = np.concatenate((indexes, bad_indexes))

    indexes = np.unique(indexes)
    return np.sort(indexes)


def convertDESCcatChunk(configfilename,data,chunknum,flag_filter_validation = True, snr_cut_validation = 5):

        """
        convertDESCcatChunk(configfilename,data,chunknum,flag_filter_validation = True, snr_cut_validation = 5)

        Convert files in ascii format to be used by Delight
        Input data can be filtered by series of filters. But it is necessary to remember which entries are kept,
        which are eliminated

        input args:
        - configfilename : Delight configuration file containing path for output files (flux variances and redshifts)
        - data : the DC2 data
        - chunknum : number of the chunk
        - filter_validation : Flag to activate quality filter data
        - snr_cut_validation : cut on flux SNR

        output :
        - the target file of the chunk which path is in configuration file
        :return:
        - the list of selected (unfiltered DC2 data)
        """
        msg="--- Convert DESC catalogs chunk {}---".format(chunknum)
        logger.info(msg)

        if FLAG_CONVERTFLUX_TODELIGHTUNIT:
            flux_multiplicative_factor = 2.22e10
        else:
            flux_multiplicative_factor = 1



        # produce a numpy array
        magdata = group_entries(data)


        # remember the number of entries
        Nin = magdata.shape[0]
        msg = "Number of objects = {} , in chunk : {}".format(Nin,chunknum)
        logger.debug(msg)


        # keep indexes to filter data with bad magnitudes
        if flag_filter_validation:
            indexes_bad_mag = filter_mag_entries(magdata)
            #magdata_f = np.delete(magdata, indexes_bad_mag, axis=0)
            magdata_f = magdata # filtering will be done later


        else:
            indexes_bad_mag=np.array([])
            magdata_f = magdata

        Nbadmag = len(indexes_bad_mag)
        msg = "Number of objects with bad magnitudes = {} , in chunk : {}".format(Nbadmag, chunknum)
        logger.debug(msg)

        #print("indexes_bad_mag = ",indexes_bad_mag)


        # convert mag to fluxes
        fdata = mag_to_flux(magdata_f)

        # keep indexes to filter data with bad SNR
        if flag_filter_validation:
            indexes_bad_snr = filter_flux_entries(fdata, nsig = snr_cut_validation)
            fdata_f = fdata
            #fdata_f = np.delete(fdata, indexes_bad, axis=0)
            #magdata_f = np.delete(magdata_f, indexes_bad, axis=0)
        else:
            fdata_f=fdata
            indexes_bad_snr = np.array([])


        Nbadsnr = len(indexes_bad_snr)
        msg = "Number of objects with bad SNR = {} , in chunk : {}".format(Nbadsnr, chunknum)
        logger.debug(msg)

        #print("indexes_bad_snr = ", indexes_bad_snr)

        # make union of indexes (unique id) before removing them for Delight
        idxToRemove = reduce(np.union1d,(indexes_bad_mag,indexes_bad_snr))
        NtoRemove=len(idxToRemove)
        msg = "Number of objects filtered out = {} , in chunk : {}".format(NtoRemove, chunknum)
        logger.debug(msg)

        #print("indexes_to_remove = ", idxToRemove)

        #pprint(idxToRemove)

        # fdata_f contains the fluxes and errors to be send to Delight

        # indexes of full input dataset
        idxInitial = np.arange(Nin)

        if NtoRemove>0:
            fdata_f = np.delete(fdata_f,idxToRemove, axis=0)
            idxFinal=np.delete(idxInitial,idxToRemove, axis=0)
        else:
            idxFinal = idxInitial


        Nkept = len(idxFinal)
        msg = "Number of objects kept = {} , in chunk : {}".format(Nkept, chunknum)
        logger.debug(msg)

        #print("indexes_kept = ", idxFinal)



        gid = fdata_f[:, 0]
        rs = fdata_f[:, 1]

        # 2) parameter file

        params = parseParamFile(configfilename, verbose=False, catFilesNeeded=False)

        numB = len(params['bandNames'])
        numObjects = len(gid)

        msg = "get {} objects ".format(numObjects)
        logger.debug(msg)

        logger.debug(params['bandNames'])

        # Generate target data
        # -------------------------

        # what is fluxes and fluxes variance
        fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))

        # loop on objects to simulate for the target and save in output trarget file
        for k in range(numObjects):
            # loop on number of bands
            for i in range(numB):
                trueFlux = fdata_f[k, 2 + i]
                noise = fdata_f[k, 8 + i]

                # put the DC2 data to the internal units of Delight
                trueFlux *= flux_multiplicative_factor
                noise *= flux_multiplicative_factor


                # fluxes[k, i] = trueFlux + noise * np.random.randn() # noisy flux
                fluxes[k, i] = trueFlux

                if fluxes[k, i] < 0:
                    # fluxes[k, i]=np.abs(noise)/10.
                    fluxes[k, i] = trueFlux

                fluxesVar[k, i] = noise ** 2.

        # container for target galaxies output
        # at some redshift, provides the flux and its variance inside each band
        

        data = np.zeros((numObjects, 1 + len(params['target_bandOrder'])))
        bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn, refBandColumn = readColumnPositions(params,
                                                                                                                 prefix="target_")

        for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
            data[:, pf] = fluxes[:, ib]
            data[:, pfv] = fluxesVar[:, ib]
        data[:, redshiftColumn] = rs
        data[:, -1] = 0  # NO TYPE

        msg = "write file {}".format(os.path.basename(params['targetFile']))
        logger.debug(msg)

        msg = "write target file {}".format(params['targetFile'])
        logger.debug(msg)

        outputdir = os.path.dirname(params['targetFile'])
        if not os.path.exists(outputdir):
            msg = " outputdir not existing {} then create it ".format(outputdir)
            logger.info(msg)
            os.makedirs(outputdir)

        np.savetxt(params['targetFile'], data)

        # return the index of selected data
        return idxFinal



def convertDESCcat(configfilename,desctraincatalogfile,desctargetcatalogfile,\
                   flag_filter_training=True,flag_filter_validation=True,snr_cut_training=5,snr_cut_validation=5):

    """
    convertDESCcat(configfilename,desctraincatalogfile,desctargetcatalogfile,\
                   flag_filter_training=True,flag_filter_validation=True,snr_cut_training=5,snr_cut_validation=5):


    Convert files in ascii format to be used by Delight

    input args:
    - configfilename : Delight configuration file containingg path for output files (flux variances and redshifts)
    - desctraincatalogfile : training file provided by RAIL (hdf5 format)
    - desctargetcatalogfile : target file provided by RAIL (hdf5 format)
    - flag_filter_training : Activate filtering on training data
    - flag_filter_validation : Activate filtering on validation data
    - snr_cut_training : Cut on flux SNR in training data
    - snr_cut_validation : Cut on flux SNR in validation data

    output :
    - the Delight training and target file which path is in configuration file

    :return: nothing

    """


    logger.info("--- Convert DESC training and target catalogs ---")

    if FLAG_CONVERTFLUX_TODELIGHTUNIT:
        flux_multiplicative_factor = 2.22e10
    else:
        flux_multiplicative_factor = 1



    # 1) DESC catalog file
    msg="read DESC hdf5 training file {} ".format(desctraincatalogfile)
    logger.debug(msg)

    f = load_raw_hdf5_data(desctraincatalogfile, groupname='photometry')

    # produce a numpy array
    magdata = group_entries(f)

    # remember the number of entries
    Nin = magdata.shape[0]
    msg = "Number of objects = {} , in  training dataset".format(Nin)
    logger.debug(msg)



    # keep indexes to filter data with bad magnitudes
    if flag_filter_training:
        indexes_bad_mag = filter_mag_entries(magdata)
        # magdata_f = np.delete(magdata, indexes_bad_mag, axis=0)
        magdata_f = magdata  # filtering will be done later
    else:
        indexes_bad_mag = np.array([])
        magdata_f = magdata

    Nbadmag = len(indexes_bad_mag)
    msg = "Number of objects with bad magnitudes {}  in training dataset".format(Nbadmag)
    logger.debug(msg)


    # convert mag to fluxes
    fdata = mag_to_flux(magdata_f)

    # keep indexes to filter data with bad SNR
    if flag_filter_training:
        indexes_bad_snr = filter_flux_entries(fdata, nsig=snr_cut_training)
        fdata_f = fdata
        # fdata_f = np.delete(fdata, indexes_bad, axis=0)
        # magdata_f = np.delete(magdata_f, indexes_bad, axis=0)
    else:
        fdata_f = fdata
        indexes_bad_snr = np.array([])

    Nbadsnr = len(indexes_bad_snr)
    msg = "Number of objects with bad SNR = {} , in  training dataset".format(Nbadsnr)
    logger.debug(msg)

    # make union of indexes (unique id) before removing them for Delight
    idxToRemove = reduce(np.union1d, (indexes_bad_mag, indexes_bad_snr))
    NtoRemove = len(idxToRemove)
    msg = "Number of objects filtered out = {} , in training dataset".format(NtoRemove)
    logger.debug(msg)


    # fdata_f contains the fluxes and errors to be send to Delight

    # indexes of full input dataset
    idxInitial = np.arange(Nin)

    if NtoRemove > 0:
        fdata_f = np.delete(fdata_f, idxToRemove, axis=0)
        idxFinal = np.delete(idxInitial, idxToRemove, axis=0)
    else:
        idxFinal = idxInitial


    Nkept = len(idxFinal)
    msg = "Number of objects kept = {} , in training dataset".format(Nkept)
    logger.debug(msg)



    gid = fdata_f[:, 0]
    rs = fdata_f[:, 1]


    # 2) parameter file

    params = parseParamFile(configfilename, verbose=False, catFilesNeeded=False)

    numB = len(params['bandNames'])
    numObjects = len(gid)

    msg = "get {} objects ".format(numObjects)
    logger.debug(msg)

    logger.debug(params['bandNames'])



    # Generate training data
    #-------------------------


    # what is fluxes and fluxes variance
    fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))

    # loop on objects to simulate for the training and save in output training file
    for k in range(numObjects):
        #loop on number of bands
        for i in range(numB):
            trueFlux = fdata_f[k,2+i]
            noise    = fdata_f[k,8+i]

            # put the DC2 data to the internal units of Delight
            trueFlux *= flux_multiplicative_factor
            noise *= flux_multiplicative_factor


            #fluxes[k, i] = trueFlux + noise * np.random.randn() # noisy flux
            fluxes[k, i] = trueFlux

            if fluxes[k, i]<0:
                #fluxes[k, i]=np.abs(noise)/10.
                fluxes[k, i] = trueFlux

            fluxesVar[k, i] = noise**2.

    # container for training galaxies output
    # at some redshift, provides the flux and its variance inside each band
    data = np.zeros((numObjects, 1 + len(params['training_bandOrder'])))
    bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,refBandColumn = readColumnPositions(params, prefix="training_")

    for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
        data[:, pf] = fluxes[:, ib]
        data[:, pfv] = fluxesVar[:, ib]
    data[:, redshiftColumn] = rs
    data[:, -1] = 0  # NO type


    msg="write training file {}".format(params['trainingFile'])
    logger.debug(msg)

    outputdir=os.path.dirname(params['trainingFile'])
    if not os.path.exists(outputdir):
        msg = " outputdir not existing {} then create it ".format(outputdir)
        logger.info(msg)
        os.makedirs(outputdir)


    np.savetxt(params['trainingFile'], data)




    # Generate Target data : procedure similar to the training
    #-----------------------------------------------------------

    # 1) DESC catalog file
    msg = "read DESC hdf5 validation file {} ".format(desctargetcatalogfile)
    logger.debug(msg)

    f = load_raw_hdf5_data(desctargetcatalogfile, groupname='photometry')

    # produce a numpy array
    magdata = group_entries(f)


    # remember the number of entries
    Nin = magdata.shape[0]
    msg = "Number of objects = {} , in  validation dataset".format(Nin)
    logger.debug(msg)


    # filter bad data
    # keep indexes to filter data with bad magnitudes
    if flag_filter_validation:
        indexes_bad_mag = filter_mag_entries(magdata)
        # magdata_f = np.delete(magdata, indexes_bad_mag, axis=0)
        magdata_f = magdata  # filtering will be done later
    else:
        indexes_bad_mag = np.array([])
        magdata_f = magdata

    Nbadmag = len(indexes_bad_mag)
    msg = "Number of objects with bad magnitudes = {} , in validation dataset".format(Nbadmag)
    logger.debug(msg)



    # convert mag to fluxes
    fdata = mag_to_flux(magdata_f)

    # keep indexes to filter data with bad SNR
    if flag_filter_validation:
        indexes_bad_snr = filter_flux_entries(fdata, nsig=snr_cut_validation)
        fdata_f = fdata
        # fdata_f = np.delete(fdata, indexes_bad, axis=0)
        # magdata_f = np.delete(magdata_f, indexes_bad, axis=0)
    else:
        fdata_f = fdata
        indexes_bad_snr = np.array([])

    Nbadsnr = len(indexes_bad_snr)
    msg = "Number of objects with bad SNR = {} , in  validation dataset".format(Nbadsnr)
    logger.debug(msg)

    # make union of indexes (unique id) before removing them for Delight
    idxToRemove = reduce(np.union1d, (indexes_bad_mag, indexes_bad_snr))
    NtoRemove = len(idxToRemove)
    msg = "Number of objects filtered out = {} , in validation dataset".format(NtoRemove)
    logger.debug(msg)

    # fdata_f contains the fluxes and errors to be send to Delight

    # indexes of full input dataset
    idxInitial = np.arange(Nin)

    if NtoRemove > 0:
        fdata_f = np.delete(fdata_f, idxToRemove, axis=0)
        idxFinal = np.delete(idxInitial, idxToRemove, axis=0)
    else:
        idxFinal = idxInitial


    Nkept = len(idxFinal)
    msg = "Number of objects kept = {} , in validation dataset".format(Nkept)
    logger.debug(msg)

    gid = fdata_f[:, 0]
    rs = fdata_f[:, 1]

    numObjects = len(gid)
    msg = "get {} objects ".format(numObjects)
    logger.debug(msg)

    fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))

    # loop on objects in target files
    for k in range(numObjects):
        # loop on bands
        for i in range(numB):
            # compute the flux in that band at the redshift
            trueFlux = fdata_f[k, 2 + i]
            noise = fdata_f[k, 8 + i]

            # put the DC2 data to the internal units of Delight
            trueFlux *= flux_multiplicative_factor
            noise *= flux_multiplicative_factor

            #fluxes[k, i] = trueFlux + noise * np.random.randn()
            fluxes[k, i] = trueFlux

            if fluxes[k, i]<0:
                #fluxes[k, i]=np.abs(noise)/10.
                fluxes[k, i] = trueFlux

            fluxesVar[k, i] = noise**2



    data = np.zeros((numObjects, 1 + len(params['target_bandOrder'])))
    bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,refBandColumn = readColumnPositions(params, prefix="target_")

    for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
        data[:, pf] = fluxes[:, ib]
        data[:, pfv] = fluxesVar[:, ib]
    data[:, redshiftColumn] = rs
    data[:, -1] = 0  # NO TYPE

    msg = "write file {}".format(os.path.basename(params['targetFile']))
    logger.debug(msg)

    msg = "write target file {}".format(params['targetFile'])
    logger.debug(msg)

    outputdir = os.path.dirname(params['targetFile'])
    if not os.path.exists(outputdir):
        msg = " outputdir not existing {} then create it ".format(outputdir)
        logger.info(msg)
        os.makedirs(outputdir)

    np.savetxt(params['targetFile'], data)

################################################################################
# New version of RAIL with data structure directly provided: (SDC 2021/10/23)  #
################################################################################

def convertDESCcatTrainData(configfilename,descatalogdata,flag_filter=True,snr_cut=5):

    """
    convertDESCcatData(configfilename,desccatalogdata,
                   flag_filter=True,snr_cut=5,s):


    Convert files in ascii format to be used by Delight

    input args:
    - configfilename : Delight configuration file containingg path for output files (flux variances and redshifts)
    - desccatalogdata : data provided by RAIL (dictionary format)
 
    - flag_filter : Activate filtering on training data

    - snr_cut: Cut on flux SNR in training data
  

    output :
    - the Delight training  which path is in configuration file

    :return: nothing

    """


    logger.info("--- Convert DESC training  catalogs data ---")

    if FLAG_CONVERTFLUX_TODELIGHTUNIT:
        flux_multiplicative_factor = 2.22e10
    else:
        flux_multiplicative_factor = 1



    # 1) DESC catalog data
    # msg="read DESC hdf5 training data "
    # logger.debug(msg)

    #f = load_raw_hdf5_data(desctraincatalogfile, groupname='photometry')

    # produce a numpy array
    #magdata = group_entries(f)
    
    logger.debug("--- descatalogdata : ---")
    pprint.pprint(descatalogdata)
    
    magdata = group_entries(descatalogdata)
    
    logger.debug("--- magdata : ---")
    pprint.pprint(magdata)
   

    # remember the number of entries
    Nin = magdata.shape[0]
    msg = "Number of objects = {} , in  training dataset".format(Nin)
    logger.debug(msg)



    # keep indexes to filter data with bad magnitudes
    if flag_filter:
        indexes_bad_mag = filter_mag_entries(magdata)
        # magdata_f = np.delete(magdata, indexes_bad_mag, axis=0)
        magdata_f = magdata  # filtering will be done later
    else:
        indexes_bad_mag = np.array([])
        magdata_f = magdata

    Nbadmag = len(indexes_bad_mag)
    msg = "Number of objects with bad magnitudes {}  in training dataset".format(Nbadmag)
    logger.debug(msg)


    # convert mag to fluxes
    fdata = mag_to_flux(magdata_f)

    # keep indexes to filter data with bad SNR
    if flag_filter:
        indexes_bad_snr = filter_flux_entries(fdata, nsig=snr_cut)
        fdata_f = fdata
        # fdata_f = np.delete(fdata, indexes_bad, axis=0)
        # magdata_f = np.delete(magdata_f, indexes_bad, axis=0)
    else:
        fdata_f = fdata
        indexes_bad_snr = np.array([])

    Nbadsnr = len(indexes_bad_snr)
    msg = "Number of objects with bad SNR = {} , in  training dataset".format(Nbadsnr)
    logger.debug(msg)

    # make union of indexes (unique id) before removing them for Delight
    idxToRemove = reduce(np.union1d, (indexes_bad_mag, indexes_bad_snr))
    NtoRemove = len(idxToRemove)
    msg = "Number of objects filtered out = {} , in training dataset".format(NtoRemove)
    logger.debug(msg)


    # fdata_f contains the fluxes and errors to be send to Delight

    # indexes of full input dataset
    idxInitial = np.arange(Nin)

    if NtoRemove > 0:
        fdata_f = np.delete(fdata_f, idxToRemove, axis=0)
        idxFinal = np.delete(idxInitial, idxToRemove, axis=0)
    else:
        idxFinal = idxInitial


    Nkept = len(idxFinal)
    msg = "Number of objects kept = {} , in training dataset".format(Nkept)
    logger.debug(msg)



    gid = fdata_f[:, 0]
    rs = fdata_f[:, 1]


    # 2) parameter file
    #-------------------

    params = parseParamFile(configfilename, verbose=False, catFilesNeeded=False)

    numB = len(params['bandNames'])
    numObjects = len(gid)

    msg = "get {} objects ".format(numObjects)
    logger.debug(msg)

    logger.debug(params['bandNames'])



    # Generate training data
    #-------------------------


    # what is fluxes and fluxes variance
    fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))

    # loop on objects to simulate for the training and save in output training file
    for k in range(numObjects):
        #loop on number of bands
        for i in range(numB):
            trueFlux = fdata_f[k,2+i]
            noise    = fdata_f[k,8+i]

            # put the DC2 data to the internal units of Delight
            trueFlux *= flux_multiplicative_factor
            noise *= flux_multiplicative_factor


            #fluxes[k, i] = trueFlux + noise * np.random.randn() # noisy flux
            fluxes[k, i] = trueFlux

            if fluxes[k, i]<0:
                #fluxes[k, i]=np.abs(noise)/10.
                fluxes[k, i] = trueFlux

            fluxesVar[k, i] = noise**2.

    # container for training galaxies output
    # at some redshift, provides the flux and its variance inside each band
    data = np.zeros((numObjects, 1 + len(params['training_bandOrder'])))
    bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,refBandColumn = readColumnPositions(params, prefix="training_")

    for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
        data[:, pf] = fluxes[:, ib]
        data[:, pfv] = fluxesVar[:, ib]
    data[:, redshiftColumn] = rs
    data[:, -1] = 0  # NO type


    msg="write training file {}".format(params['trainingFile'])
    logger.debug(msg)

    outputdir=os.path.dirname(params['trainingFile'])
    if not os.path.exists(outputdir):
        msg = " outputdir not existing {} then create it ".format(outputdir)
        logger.info(msg)
        os.makedirs(outputdir)


    np.savetxt(params['trainingFile'], data)

#---

def convertDESCcatTargetFile(configfilename,desctargetcatalogfile,flag_filter=True,snr_cut=5):

    """
    convertDESCcatTargetFile(configfilename,desctargetcatalogfile,flag_filter=True,snr_cut)
    

    Convert files in ascii format to be used by Delight

    input args:
    - configfilename : Delight configuration file containingg path for output files (flux variances and redshifts)
    - desctargetcatalogfile : target file provided by RAIL (hdf5 format)
    - flag_filter_ : Activate filtering on validation data
    - snr_cut: Cut on flux SNR in validation data

    output :
    - the Delight target file which path is in configuration file

    :return: nothing

    """


    logger.info("--- Convert DESC target catalogs ---")

    if FLAG_CONVERTFLUX_TODELIGHTUNIT:
        flux_multiplicative_factor = 2.22e10
    else:
        flux_multiplicative_factor = 1



    # Generate Target data : procedure similar to the training
    #-----------------------------------------------------------

    # 1) DESC catalog file
    #---------------------
    
    msg = "read DESC hdf5 validation file {} ".format(desctargetcatalogfile)
    logger.debug(msg)

    f = load_raw_hdf5_data(desctargetcatalogfile, groupname='photometry')

    # produce a numpy array
    magdata = group_entries(f)


    # remember the number of entries
    Nin = magdata.shape[0]
    msg = "Number of objects = {} , in  validation dataset".format(Nin)
    logger.debug(msg)


    # filter bad data
    # keep indexes to filter data with bad magnitudes
    if flag_filter:
        indexes_bad_mag = filter_mag_entries(magdata)
        # magdata_f = np.delete(magdata, indexes_bad_mag, axis=0)
        magdata_f = magdata  # filtering will be done later
    else:
        indexes_bad_mag = np.array([])
        magdata_f = magdata

    Nbadmag = len(indexes_bad_mag)
    msg = "Number of objects with bad magnitudes = {} , in validation dataset".format(Nbadmag)
    logger.debug(msg)



    # convert mag to fluxes
    fdata = mag_to_flux(magdata_f)

    # keep indexes to filter data with bad SNR
    if flag_filter:
        indexes_bad_snr = filter_flux_entries(fdata, nsig=snr_cut)
        fdata_f = fdata
        # fdata_f = np.delete(fdata, indexes_bad, axis=0)
        # magdata_f = np.delete(magdata_f, indexes_bad, axis=0)
    else:
        fdata_f = fdata
        indexes_bad_snr = np.array([])

    Nbadsnr = len(indexes_bad_snr)
    msg = "Number of objects with bad SNR = {} , in  validation dataset".format(Nbadsnr)
    logger.debug(msg)

    # make union of indexes (unique id) before removing them for Delight
    idxToRemove = reduce(np.union1d, (indexes_bad_mag, indexes_bad_snr))
    NtoRemove = len(idxToRemove)
    msg = "Number of objects filtered out = {} , in validation dataset".format(NtoRemove)
    logger.debug(msg)

    # fdata_f contains the fluxes and errors to be send to Delight

    # indexes of full input dataset
    idxInitial = np.arange(Nin)

    if NtoRemove > 0:
        fdata_f = np.delete(fdata_f, idxToRemove, axis=0)
        idxFinal = np.delete(idxInitial, idxToRemove, axis=0)
    else:
        idxFinal = idxInitial


    Nkept = len(idxFinal)
    msg = "Number of objects kept = {} , in validation dataset".format(Nkept)
    logger.debug(msg)

    gid = fdata_f[:, 0]
    rs = fdata_f[:, 1]
    
    
    
    # 2) parameter file
    #------------------- 

    params = parseParamFile(configfilename, verbose=False, catFilesNeeded=False)

    numB = len(params['bandNames'])
    numObjects = len(gid)

    msg = "get {} objects ".format(numObjects)
    logger.debug(msg)

    logger.debug(params['bandNames'])
    
    
    # 3) Generate target data
    #------------------------

    numObjects = len(gid)
    msg = "get {} objects ".format(numObjects)
    logger.debug(msg)

    fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))

    # loop on objects in target files
    for k in range(numObjects):
        # loop on bands
        for i in range(numB):
            # compute the flux in that band at the redshift
            trueFlux = fdata_f[k, 2 + i]
            noise = fdata_f[k, 8 + i]

            # put the DC2 data to the internal units of Delight
            trueFlux *= flux_multiplicative_factor
            noise *= flux_multiplicative_factor

            #fluxes[k, i] = trueFlux + noise * np.random.randn()
            fluxes[k, i] = trueFlux

            if fluxes[k, i]<0:
                #fluxes[k, i]=np.abs(noise)/10.
                fluxes[k, i] = trueFlux

            fluxesVar[k, i] = noise**2


            

            

    data = np.zeros((numObjects, 1 + len(params['target_bandOrder'])))
    bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,refBandColumn = readColumnPositions(params, prefix="target_")

    for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
        data[:, pf] = fluxes[:, ib]
        data[:, pfv] = fluxesVar[:, ib]
    data[:, redshiftColumn] = rs
    data[:, -1] = 0  # NO TYPE

    msg = "write file {}".format(os.path.basename(params['targetFile']))
    logger.debug(msg)

    msg = "write target file {}".format(params['targetFile'])
    logger.debug(msg)

    outputdir = os.path.dirname(params['targetFile'])
    if not os.path.exists(outputdir):
        msg = " outputdir not existing {} then create it ".format(outputdir)
        logger.info(msg)
        os.makedirs(outputdir)

    np.savetxt(params['targetFile'], data)

    

if __name__ == "__main__":
    # execute only if run as a script


    msg="Start convertDESCcat.py"
    logger.info(msg)
    logger.info("--- convert DESC catalogs ---")



    if len(sys.argv) < 4:
        raise Exception('Please provide a parameter file and the training and validation and catalog files')

    convertDESCcat(sys.argv[1],sys.argv[2],sys.argv[3])
