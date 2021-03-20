"""
Example code that just spits out random numbers between 0 and 3
for z_mode, and Gaussian centered at z_mode with width
random_width*(1+zmode).
"""

import numpy as np
from scipy.stats import norm
from rail.estimation.estimator import Estimator as BaseEstimation

import os
import errno

import coloredlogs
import logging
from pkg_resources import resource_filename

from interfaces.rail.processFilters import processFilters  # interface added into delight in branch rail
#from interfaces.rail.makeConfigParam import makeConfigParam  # build the parameter file required by Delight
from interfaces.rail.makeConfigParam import *  # build the parameter file required by Delight
from interfaces.rail.processSEDs import processSEDs  # build a redshift -flux grid model
from interfaces.rail.templateFitting import templateFitting
from interfaces.rail.simulateWithSEDs import simulateWithSEDs # simulate its own SED in tutorial mode
from interfaces.rail.delightLearn import delightLearn
from interfaces.rail.delightApply import delightApply
from interfaces.rail.convertDESCcat  import convertDESCcat   # convert DESC input file into Delight format
from interfaces.rail.convertDESCcat  import *   # convert DESC input file into Delight format
from interfaces.rail.calibrateTemplateMixturePriors import *

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

        self.width = inputs['rand_width']
        self.zmin = inputs['rand_zmin']
        self.zmax = inputs['rand_zmax']
        self.nzbins = inputs['nzbins']
        self.tempdir = inputs['tempdir']
        self.tempdatadir = inputs['tempdatadir']
        self.delightparamfile=inputs["delightparamfile"]
        self.delightparamfile = os.path.join(self.tempdir, self.delightparamfile)
        self.tutorialmode = inputs["dlght_tutorialmode"]
        self.dlght_calibrateTemplateMixturePrior =inputs["dlght_calibrateTemplateMixturePrior"]
        self.tutorialpasseval = False
        self.applypassonce = False
        self.chunknum=0
        self.inputs=inputs



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




        # Very awfull way to get the internal data path for Delight
        # This is the delight setup.py tools that get the data there
        basedelight_datapath = resource_filename('delight', '../data')
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
            convertDESCcat(self.delightparamfile, self.trainfile, self.testfile,flag_filter=True)

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

        msg = " chunk number {} ".format(self.chunknum)
        logger.info(msg)

        basedelight_datapath = resource_filename('delight', '../data')
        print(self.delightparamfile)


        # when Delight runs in tutorial mode call only once delightApply
        if  self.tutorialmode and not self.tutorialpasseval:
            # Template Fitting
            templateFitting(self.delightparamfile)

            # Gaussian process fitting
            delightApply(self.delightparamfile)
            self.tutorialpasseval = True    # avoid latter call to delightApply when running in tutorial mode

        elif self.applypassonce: # case whe one want to run the whole validation dataset

            # Template Fitting
            templateFitting(self.delightparamfile)

            # TBI later with DESC data
            delightApply(self.delightparamfile)
            self.applypassonce = False

        else: # let rail split the test data into chunks
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
            convertDESCcatChunk(delightparamfilechunk, test_data, self.chunknum,flag_filter=True)

            # template fitting for that chunk
            templateFitting(delightparamfilechunk)

            # estimation for that chunk
            delightApply(delightparamfilechunk)

        pdf = []
        # allow for either format for now
        try:
            d = test_data['i_mag']
        except Exception:
            d = test_data['mag_i_lsst']
        numzs = len(d)
        zmode = np.round(np.random.uniform(0.0, self.zmax, numzs), 3)
        widths = self.width * (1.0 + zmode)
        self.zgrid = np.linspace(0., self.zmax, self.nzbins)

        for i in range(numzs):
            pdf.append(norm.pdf(self.zgrid, zmode[i], widths[i]))
        pz_dict = {'zmode': zmode, 'pz_pdf': pdf}
        return pz_dict
