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



# Create a logger object.
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s %(name)s[%(process)d] %(levelname)s %(message)s')




def_param = dict(run_params = dict(dlght_redshiftMin=0.01,
                                   dlght_redshiftMax=3.01,
                                   dlght_redshiftNumBinsGPpred = 301,
                                   nzbins = 301,
                                   dlght_redshiftBinSize = 0.01,
                                   dlght_redshiftDisBinSize = 0.2,
                                   bands_names = "DC2LSST_u DC2LSST_g DC2LSST_r DC2LSST_i DC2LSST_z DC2LSST_y",
                                   bands_path = "./rail/estimation/data/FILTER",
                                   bands_fmt = "res",
                                   bands_numcoefs = 15,
                                   bands_verbose = True,
                                   bands_makeplots = False,
                                   bands_debug = True,
                                   tempdir = "./examples/estimation/tmp",
                                   tempdatadir = "./examples/estimation/tmp/delight_data",
                                   sed_path = "./rail/estimation/data/SED",
                                   sed_name_list = "El_B2004a Sbc_B2004a Scd_B2004a SB3_B2004a SB2_B2004a Im_B2004a ssp_25Myr_z008 ssp_5Myr_z008",
                                   sed_fmt = "sed",
                                   prior_t_list = "0.27 0.26 0.25 0.069 0.021 0.11 0.0061 0.0079",
                                   prior_zt_list = "0.23 0.39 0.33 0.31 1.1 0.34 1.2 0.14",
                                   lambda_ref = 4500.,
                                   train_refbandorder = "DC2LSST_u DC2LSST_u_var DC2LSST_g DC2LSST_g_var DC2LSST_r DC2LSST_r_var DC2LSST_i DC2LSST_i_var DC2LSST_z DC2LSST_z_var DC2LSST_y DC2LSST_y_var redshift",
                                   train_refband = "DC2LSST_i",
                                   train_fracfluxerr = 1.e-4,
                                   train_xvalidate = False,
                                   train_xvalbandorder = "_ _ _ _ DC2LSST_r DC2LSST_r_var _ _ _ _ _ _",
                                   target_refbandorder = "DC2LSST_u DC2LSST_u_var DC2LSST_g DC2LSST_g_var DC2LSST_r DC2LSST_r_var DC2LSST_i DC2LSST_i_var DC2LSST_z DC2LSST_z_var DC2LSST_y DC2LSST_y_var redshift",
                                   target_refband = "DC2LSST_r",
                                   target_fracfluxerr = 1.e-4,
                                   delightparamfile = "parametersTest.cfg",
                                   dlght_tutorialmode = False,
                                   flag_filter_training = True,
                                   snr_cut_training = 5,
                                   flag_filter_validation = True,
                                   snr_cut_validation = 3,
                                   dlght_calibrateTemplateMixturePrior = False,
                                   dlght_inputdata = "./examples/estimation/tmp/delight_indata",
                                   zPriorSigma = 0.2,
                                   ellPriorSigma = 0.5,
                                   fluxLuminosityNorm = 1.0,
                                   alpha_C = 1.0e3,
                                   V_C = 0.1,
                                   alpha_L = 1.0e2,
                                   V_L = 0.1,
                                   lineWidthSigma = 20,
                                   inform_options = dict(save_train = False,
                                                         load_model = False,
                                                         modelfile = 'model.out'),
                                   
))


desc_dict = dict(dlght_redshiftMin = "min redshift",
                 dlght_redshiftMax = "max redshift",
                 dlght_redshiftNumBinsGPpred = "num bins",
                 dlght_redshiftDisBinSize = "???",
                 dlght_redshiftBinSize = "bad, shouldn't be here",
                 nzbins = "num bins",
                 bands_names = "string with list of Filter names",
                 bands_path = "string specifying path to filter directory",
                 bands_fmt = "string giving the file extension of the filters, not including the '.'",
                 bands_numcoefs = "integer specifying number of coefs in approximation of filter",
                 bands_verbose = "boolean filter verbosity",
                 bands_makeplots = "boolean for whether to make plot showing approximate filters",
                 bands_debug = "boolean debug flag for filters",
                 tempdir = "temp dir",
                 tempdatadir = "temp data dir",
                 sed_path = "path to seds",
                 sed_name_list = "String with list of all SED names, with no file extension",
                 sed_fmt = "file extension of SED files (withough the '.', e.g dat or sed",
                 prior_t_list = "String of numbers specifying prior type fracs MUST BE SAME LENGTH AS NUMBER OF SEDS",
                 prior_zt_list = "string of numbers for redshift prior, MUST BE SAME LENGTH AS NUMBER OF SEDS",
                 lambda_ref = "reference wavelength",
                 train_refbandorder = "order of bands in training?",
                 train_refband = "string name of ref band",
                 train_fracfluxerr = "float: frac err to add to flux?",
                 train_xvalidate = "bool: cross validate flag",
                 train_xvalbandorder = "Str: cols to use in cross validation?",
                 target_refbandorder = "Str: order of reference bands for target data?",
                 target_refband = "Str: the reference band for the taret data?",
                 target_fracfluxerr = "float: extra fractional error to add to target fluxes?",
                 delightparamfile = "param file",
                 dlght_tutorialmode = "bool: run in tutorial mode",
                 flag_filter_training = "bool: ?",
                 snr_cut_training = "SNR training cut",
                 flag_filter_validation = "bool: ?",
                 snr_cut_validation = "SNR val cut",
                 dlght_calibrateTemplateMixturePrior = "bool, ?",
                 dlght_inputdata = "data dir",
                 zPriorSigma = "prior thing",
                 ellPriorSigma = "prior thing",
                 fluxLuminosityNorm = "prior thing",
                 alpha_C = "prior thing",
                 V_C = "prior thing",
                 alpha_L = "prior thing",
                 V_L = "prior thing",
                 lineWidthSigma = "prior thing",
                 inform_options = "inform options")


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

        self.zmin = inputs['dlght_redshiftMin']
        if self.zmin <= 0.:
            raise ValueError("zmin must be greater than zero!"
                             + "set dlght_redshiftMin accordingly")
        self.zmax = inputs['dlght_redshiftMax']
        self.width = inputs['dlght_redshiftBinSize']
        self.zgrid = np.arange(self.zmin, self.zmax, self.width) # this is how createGrids defines the grid
        self.nzbins = len(self.zgrid)
        
        # temporary directories for Delight temprary file
        self.tempdir = inputs['tempdir']
        self.tempdatadir = inputs['tempdatadir']
        self.sed_path = inputs['sed_path']
        self.bands_path = inputs['bands_path']
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

        self.sed_path = inputs['sed_path']
        self.sed_name_list = inputs['sed_name_list']
        self.sed_fmt = inputs['sed_fmt']
        self.prior_t_list = inputs['prior_t_list']
        self.prior_zt_list = inputs['prior_zt_list']
        self.lambda_ref = inputs['lambda_ref']
        

        # counter on the chunk validation dataset
        self.chunknum=0

        self.dlght_calibrateTemplateMixturePrior = inputs["dlght_calibrateTemplateMixturePrior"]
        # all parameter files
        self.inputs = inputs

        np.random.seed(87)
        
        
        
        
        
        

    def inform(self, training_data):
        """
          this is delightPZ
        """
        
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

        #if not os.path.exists(self.delightindata):
        #    msg = " No Delight input data in dir  " + self.delightindata
        #    logger.error(msg)
        #    exit(-1)

        #SUBDIRs = ['BROWN_SEDs', 'CWW_SEDs', 'FILTERS']

        #for subdir in SUBDIRs:
        #    theinpath=os.path.join(self.delightindata,subdir)
        #    if not os.path.exists(theinpath):
        #        msg = " No Delight input data in dir  " + theinpath
        #        logger.error(msg)
        #        exit(-1)
        if not os.path.exists(self.sed_path):
            msg = " No Delight SED data in dir " + self.sed_path
            logerr.error(msg)
            exit(-1)
        if not os.path.exists(self.bands_path):
            msg = " No Delight FILTER data in dir " + self.bands_path
            logerr.error(msg)
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
        
        print("\n\n\n Starting estimation...\n\n\n")
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
            paramfile_txt=makeConfigParam(basedelight_datapath, self.inputs, self.chunknum)

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

        if self.tutorialmode:
            # fill creazy values (simulation not send to rail)
            zmode = np.round(np.random.uniform(self.zmin, self.zmax, numzs), 3)
            pdfs = np.zeros([numzs, len(self.zgrid)])
            for i in range(numzs):
                pdfs[i] = (norm.pdf(self.zgrid, zmode[i], self.width * (1 + zmode[i])))

        else:
            zmode, pdfs = \
            getDelightRedshiftEstimation(delightparamfilechunk,self.chunknum,numzs,indexes_sel)
            zmode = np.round(zmode,3)


        if self.output_format == 'qp':
            qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid,
                                                    yvals=pdfs))
            return qp_d
        else:
            pz_dict = {'zmode': zmode, 'pz_pdf': pdfs}
            return pz_dict
