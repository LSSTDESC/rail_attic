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
# Last update   : October 21th 2021
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

# Filters and SED
from rail.estimation.algos.include_delightPZ.processFilters import processFilters
from rail.estimation.algos.include_delightPZ.processSEDs import processSEDs  # build a redshift -flux grid model

# interface with Delight through files
from rail.estimation.algos.include_delightPZ.makeConfigParam import *  # build the parameter file required by Delight

# Delight format
from rail.estimation.algos.include_delightPZ.convertDESCcat  import *   # convert DESC input file into Delight format

# mock data simulation
from rail.estimation.algos.include_delightPZ.simulateWithSEDs import simulateWithSEDs # simulate its own SED in tutorial mode

# Delight algorithms

from rail.estimation.algos.include_delightPZ.templateFitting import templateFitting
from rail.estimation.algos.include_delightPZ.delightLearn import delightLearn
from rail.estimation.algos.include_delightPZ.delightApply import delightApply

# other

#from rail.estimation.algos.include_delightPZ.calibrateTemplateMixturePriors import *
from rail.estimation.algos.include_delightPZ.getDelightRedshiftEstimation import *

# Create a logger object.
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s %(name)s[%(process)d] %(levelname)s %(message)s')





import numpy as np
from scipy.stats import norm
from rail.estimation.estimator import Estimator as BaseEstimation
from rail.estimation.utils import check_and_print_params
import qp
import pprint



# Create a logger object.
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s %(name)s[%(process)d] %(levelname)s %(message)s')




def_param = {'run_params': {'rand_width': 0.025, 'rand_zmin': 0.0,
                            'rand_zmax': 3.0, 'nzbins': 301,
                            'inform_options': {'save_train': False,
                                               'load_model': False,
                                               'modelfile':
                                               'randommodel.pkl'
                                               }}}


desc_dict = {'rand_width': "rand_width (float): ad hock width of PDF",
             'rand_zmin': "rand_zmin (float): min value for z grid",
             'rand_zmax': "rand_zmax (float): max value for z grid",
             'nzbins': "nzbins (int): number of z bins",
             'inform_options': "inform_options: (dict): a "
             "dictionary of options for loading and storing of "
             "the pretrained model.  This includes:\n "
             "modelfile:(str) the filename to save or load a "
             "trained model from.\n save_train:(bool) boolean to "
             "set whether to save a trained model.\n "
             "load_model:(bool): boolean to set whether to "
             "load a trained model from filename modelfile"
             }








class delightPZ(BaseEstimation):

    def __init__(self, base_config, config_dict='None'):
        """
        Parameters
        ----------
        run_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """
        if config_dict == "None":
            print("No config file supplied, using default parameters")
            config_dict = def_param
        config_dict = check_and_print_params(config_dict, def_param,desc_dict)
        
        
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
        
        
        
        
        
        

    def inform(self, training_data):
        """
          this is delightPZ, so does nothing
        """
        print("I don't need to train!!!")
        pprint.pprint(training_data)
        
        
        
        
        msg = " INFORM "
        logger.info(msg)


        logger.info("Try to workout filters")
        
        # create usefull tempory directory
        # --------------------------------
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
        #basedelight_datapath = resource_filename('delight', '../data')
        basedelight_datapath = self.delightindata

        
          
        # create parameter file
        
        logger.debug("Guessed basedelight_datapath  = " + basedelight_datapath )

        # gives to Delight where are its datapath and provide yaml arguments
        paramfile_txt=makeConfigParam(basedelight_datapath, self.inputs)
        logger.debug(paramfile_txt)


        # save the config  parameter file that Delight needs
        with open(self.delightparamfile, 'w') as out:
            out.write(paramfile_txt)
            

            
        # The data required by Delight : SED and Filters 
        # For the moment, there is no automatic installation inside RAIL
        # These must be installed by the user, later these will be copied from Delight installation automatically

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
                
                
        # Initialisation of Delight with 1) Filters 2) SED to get Flux-redshift model 

        # Build LSST filter model
        processFilters(self.delightparamfile)

        # Build its own LSST-Flux-Redshift Model
        processSEDs(self.delightparamfile)

        if self.tutorialmode:
            # Delight build its own Mock simulations
            simulateWithSEDs(self.delightparamfile)


        else:  # convert training_data  into ascii in desc input mode
            convertDESCcatTrainData(self.delightparamfile,\
                                    training_data,flag_filter=self.flag_filter_training,\
                                    snr_cut=self.snr_cut_training)
            
            # convert target Files into ascii 
            # Delight need to know this target file
            convertDESCcatTargetFile(self.delightparamfile,self.testfile,\
                                flag_filter=self.flag_filter_validation,\
                                     snr_cut=self.snr_cut_validation)
            
            if self.dlght_calibrateTemplateMixturePrior:
                calibrateTemplateMixturePriors(self.delightparamfile)
        

        # Learn with Gaussian processes
        delightLearn(self.delightparamfile)
        
        logger.info("End of Inform")
        

    def load_pretrained_model(self):
        pass

    def estimate(self, test_data):
        
        
        self.chunknum += 1

        msg = " ESTIMATE : chunk number {} ".format(self.chunknum)
        logger.info(msg)

    
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
            zmode,widths = \
            getDelightRedshiftEstimation(delightparamfilechunk,self.chunknum,numzs,indexes_sel)
            zmode = np.round(zmode,3)


        for i in range(numzs):
            pdf.append(norm.pdf(self.zgrid, zmode[i], widths[i]))
        if self.output_format == 'qp':
            qp_d = qp.Ensemble(qp.stats.norm, data=dict(loc=zmode,
                                                        scale=widths))
            return qp_d
        else:

            pz_dict = {'zmode': zmode, 'pz_pdf': pdf}

            return pz_dict
