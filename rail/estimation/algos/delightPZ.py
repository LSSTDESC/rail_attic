##########################################################################################
#
# RAIL interface class to Delight
#
# Steering Delight from RAIL
# Used on Vera C. Runbin LSST only estimation
#
# Author        : Sylvie Dagoret-Campagne
# Affiliation   : IJCLab/IN2P3/CNRS/France
# Creation date : March 2021
# Last update   : March 21th 2021
#
############################################################################################

import numpy as np
from scipy.stats import norm
from rail.estimation.estimator import Estimator as BaseEstimation

import os
import errno

import coloredlogs
import logging
from pkg_resources import resource_filename

# Delight initialisation

#from interfaces.rail.processFilters import processFilters  # interface added into delight in branch rail
from rail.estimation.algos.include_delightPZ.processFilters import processFilters
#from interfaces.rail.processSEDs import processSEDs  # build a redshift -flux grid model
from rail.estimation.algos.include_delightPZ.processSEDs import processSEDs  # build a redshift -flux grid model

# interface with Delight through files
#from interfaces.rail.makeConfigParam import *  # build the parameter file required by Delight
from rail.estimation.algos.include_delightPZ.makeConfigParam import *  # build the parameter file required by Delight

#from interfaces.rail.convertDESCcat  import convertDESCcat   # convert DESC input file into Delight format
#from interfaces.rail.convertDESCcat  import *   # convert DESC input file into Delight format
#from from rail.estimation.algos.include_delightPZ.convertDESCcat  import convertDESCcat   # convert DESC input file into Delight format
from rail.estimation.algos.include_delightPZ.convertDESCcat  import *   # convert DESC input file into Delight format

# mock data simulation
#from interfaces.rail.simulateWithSEDs import simulateWithSEDs # simulate its own SED in tutorial mode
from rail.estimation.algos.include_delightPZ.simulateWithSEDs import simulateWithSEDs # simulate its own SED in tutorial mode

# Delight algorithms

#from interfaces.rail.templateFitting import templateFitting
from rail.estimation.algos.include_delightPZ.templateFitting import templateFitting
#from interfaces.rail.delightLearn import delightLearn
from rail.estimation.algos.include_delightPZ.delightLearn import delightLearn
#from interfaces.rail.delightApply import delightApply
from rail.estimation.algos.include_delightPZ.delightApply import delightApply

# other

#from interfaces.rail.calibrateTemplateMixturePriors import *
from rail.estimation.algos.include_delightPZ.calibrateTemplateMixturePriors import *
#from interfaces.rail.getDelightRedshiftEstimation import *
from rail.estimation.algos.include_delightPZ.getDelightRedshiftEstimation import *

# Create a logger object.
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s %(name)s[%(process)d] %(levelname)s %(message)s')


class delightPZ(BaseEstimation):

    def __init__(self, base_config, config_dict):
        """
        Parameters:
        -----------
        run_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """
        super().__init__(base_config=base_config, config_dict=config_dict)

        inputs = self.config_dict['run_params']

        self.width = inputs['dlght_redshiftBinSize']
        self.zmin = inputs['dlght_redshiftMin']
        self.zmax = inputs['dlght_redshiftMax']
        self.nzbins = int((self.zmax-self.zmin)/self.width)

        # temporary directories for Delight temprary file
        self.tempdir = inputs['tempdir']
        self.tempdatadir = inputs['tempdatadir']
        # name of delight configuration file
        self.delightparamfile=inputs["delightparamfile"]
        self.delightparamfile = os.path.join(self.tempdir, self.delightparamfile)

        self.delightindata=inputs['dlght_inputdata']

        # Choice of the running mode
        self.tutorialmode = inputs["dlght_tutorialmode"]
        self.tutorialpasseval = False  # onmy one chunk for simulation
        # for standard mode with DC2 dataset
        self.flag_filter_training = inputs["flag_filter_training"]
        self.snr_cut_training = inputs["snr_cut_training"]
        self.flag_filter_validation = inputs["flag_filter_validation"]
        self.snr_cut_validation = inputs["snr_cut_validation"]


        # counter on the chunk validation dataset
        self.chunknum=0

        self.dlght_calibrateTemplateMixturePrior = inputs["dlght_calibrateTemplateMixturePrior"]
        # all parameter files
        self.inputs = inputs

        np.random.seed(87)

    #############################################################################################################
    #
    # INFORM
    #
    ############################################################################################################

    def inform(self):
        """
          this is random, so does nothing
        """

        msg = " INFORM "
        logger.info(msg)


        logger.info("Try to workout filters")

        # create usefull tempory directory
        try:
            if not os.path.exists(self.tempdir):
                os.makedirs(self.tempdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                msg = "error creating file "+self.tempdir
                logger.error(msg)
                raise

        try:
            if not os.path.exists(self.tempdatadir):
                os.makedirs(self.tempdatadir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                msg = "error creating file " + self.tempdatadir
                logger.error(msg)
                raise


        if not os.path.exists(self.delightindata):
            msg = " No Delight input data in dir  " + self.delightindata
            logger.error(msg)
            exit(-1)

        SUBDIRs = ['BROWN_SEDs', 'CWW_SEDs', 'FILTERS']

        for subdir in SUBDIRs:
            theinpath=os.path.join(self.delightindata,subdir)
            if not os.path.exists(theinpath):
                msg = " No Delight input data in dir  " + theinpath
                logger.error(msg)
                exit(-1)








        # Very awfull way to get the internal data path for Delight
        # This is the delight setup.py tools that get the data there
        #basedelight_datapath = resource_filename('delight', '../data')
        basedelight_datapath = self.delightindata

        logger.debug("Guessed basedelight_datapath  = " + basedelight_datapath )

        # gives to Delight where are its datapath and provide yaml arguments
        paramfile_txt=makeConfigParam(basedelight_datapath, self.inputs)
        logger.debug(paramfile_txt)


        # save the config  parameter file that Delight needs
        with open(self.delightparamfile, 'w') as out:
            out.write(paramfile_txt)


        # Later will steer the calls to the Delight functions with RAIL config file

        # Build LSST filter model
        processFilters(self.delightparamfile)

        # Build its own LSST-Flux-Redshift Model
        processSEDs(self.delightparamfile)

        if self.tutorialmode:
            # Delight build its own Mock simulations
            simulateWithSEDs(self.delightparamfile)

        else:  # convert hdf5 into ascii in desc input mode
            convertDESCcat(self.delightparamfile, self.trainfile, self.testfile,\
                           flag_filter_training=self.flag_filter_training,\
                           flag_filter_validation=self.flag_filter_validation,\
                           snr_cut_training=self.snr_cut_training,\
                           snr_cut_validation=self.snr_cut_validation)


            if self.dlght_calibrateTemplateMixturePrior:
                calibrateTemplateMixturePriors(self.delightparamfile)




        # Learn with Gaussian processes
        delightLearn(self.delightparamfile)


    #############################################################################################################
    #
    # ESTIMATE
    #
    ############################################################################################################


    def estimate(self, test_data):

        self.chunknum += 1

        msg = " ESTIMATE : chunk number {} ".format(self.chunknum)
        logger.info(msg)

        #basedelight_datapath = resource_filename('delight', '../data')
        basedelight_datapath = self.delightindata

        msg=" Delight input data file are in dir : {} ".format(self.delightparamfile)
        logger.debug(msg)

        # when Delight runs in tutorial mode call only once delightApply
        if  self.tutorialmode and not self.tutorialpasseval:

            msg = "TUTORIAL MODE : process chunk {}".format(self.chunknum)
            logger.info(msg)

            # Template Fitting
            templateFitting(self.delightparamfile)

            # Gaussian process fitting
            delightApply(self.delightparamfile)
            self.tutorialpasseval = True    # avoid latter call to delightApply when running in tutorial mode

        elif self.tutorialmode and self.tutorialpasseval:
            msg="TUTORIAL MODE : skip chunk {}".format(self.chunknum)
            logger.info(msg)

        elif not self.tutorialmode : # let rail split the test data into chunks
            msg = "STANDARD MODE : process chunk {}".format(self.chunknum)
            logger.info(msg)

            # Generate a new parameter file for delight this chunk
            paramfile_txt=makeConfigParamChunk(basedelight_datapath, self.inputs, self.chunknum)

            # generate the config-parameter filename from chunk number
            delightparamfile=self.delightparamfile
            logger.debug(delightparamfile)
            dirn=os.path.dirname(delightparamfile)
            basn=os.path.basename(delightparamfile)
            basnsplit=basn.split(".")
            basnchunk =  basnsplit[0] + "_" + str(self.chunknum) + "." + basnsplit[1]
            delightparamfilechunk = os.path.join(dirn,basnchunk)
            logger.debug("parameter file for delight :" + delightparamfilechunk)

            # save the config parameter file for the data chunk that Delight needs
            with open(delightparamfilechunk, 'w') as out:
                out.write(paramfile_txt)

            # convert the chunk data into the required  flux-redshift validation file for delight
            indexes_sel=convertDESCcatChunk(delightparamfilechunk, test_data, self.chunknum,\
                                flag_filter_validation=self.flag_filter_validation,\
                                snr_cut_validation=self.snr_cut_validation)

            # template fitting for that chunk
            templateFitting(delightparamfilechunk)

            # estimation for that chunk
            delightApply(delightparamfilechunk)



        else:
            msg = "STANDARD MODE : pass chunk {}".format(self.chunknum)
            logger.info(msg)


        pdf = []
        # allow for either format for now
        try:
            d = test_data['i_mag']
        except Exception:
            d = test_data['mag_i_lsst']

        numzs = len(d)
        self.zgrid = np.linspace(self.zmin, self.zmax, self.nzbins)


        if self.tutorialmode:
            # fill creazy values (simulation not send to rail)
            zmode = np.round(np.random.uniform(self.zmax, self.zmax, numzs), 3)
            widths = self.width * (1.0 + zmode)

        else:
            zmode,widths = getDelightRedshiftEstimation(delightparamfilechunk,self.chunknum,numzs,indexes_sel)
            zmode = np.round(zmode,3)


        for i in range(numzs):
            pdf.append(norm.pdf(self.zgrid, zmode[i], widths[i]))

        pz_dict = {'zmode': zmode, 'pz_pdf': pdf}

        return pz_dict
