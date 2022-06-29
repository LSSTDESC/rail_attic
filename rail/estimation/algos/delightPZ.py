"""

RAIL interface class to Delight

Steering Delight from RAIL
Used on Vera C. Rubin LSST only estimation

Author        : Sylvie Dagoret-Campagne, Sam Schmidt, others
Affiliation   : IJCLab/IN2P3/CNRS/France
Creation date : March 2021
Last update   : October 21th 2021
Last update   : February 25th 2022
"""

import sys
import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.data import TableHandle

import qp

import os
import errno

import coloredlogs
import logging

# Filters and SED

# Create a logger object.
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s,%(msecs)03d %(programname)s %(name)s[%(process)d] %(levelname)s %(message)s')


class Inform_DelightPZ(CatInformer):
    """Train the Delight code, outputs are actually saved to files,
    which is fairly non-standard way currently
    """
    name = 'Inform_DelightPZ'
    outputs = []
    config_options = CatInformer.config_options.copy()
    config_options.update(dlght_redshiftMin=Param(float, 0.01, msg='min redshift'),
                          dlght_redshiftMax=Param(float, 3.01, msg='max redshift'),
                          dlght_redshiftNumBinsGPpred=Param(int, 301, msg='num bins'),
                          nzbins=Param(int, 301, msg="num z bins"),
                          dlght_redshiftBinSize=Param(float, 0.01, msg='???'),
                          dlght_redshiftDisBinSize=Param(float, 0.2, msg='bad, shouldnt be here'),
                          bands_names=Param(str, "DC2LSST_u DC2LSST_g DC2LSST_r DC2LSST_i DC2LSST_z DC2LSST_y", msg='string with list of Filter names'),
                          bands_path=Param(str, "./examples/estimation/data/FILTER", msg='string specifying path to filter directory'),
                          bands_fmt=Param(str, "res", msg="string giving the file extension of the filters, not including the '.'"),
                          bands_numcoefs=Param(int, 15, msg='integer specifying number of coefs in approximation of filter'),
                          bands_verbose=Param(bool, True, msg='verbose'),
                          bands_makeplots=Param(bool, False, msg='bool for whether to make approx band plots'),
                          bands_debug=Param(bool, True, msg='debug flag for filters'),
                          tempdir=Param(str, "./examples/estimation/tmp", msg='temp dir'),
                          tempdatadir=Param(str, "./examples/estimation/tmp/delight_data", msg='temp data dir'),
                          sed_path=Param(str, "./examples/estimation/data/SED", msg='path to SED dir'),
                          sed_name_list=Param(str, "El_B2004a Sbc_B2004a Scd_B2004a SB3_B2004a SB2_B2004a Im_B2004a ssp_25Myr_z008 ssp_5Myr_z008", msg='String with list of all SED names, with no file extension'),
                          sed_fmt=Param(str, "sed", msg="file extension of SED files (withough the '.', e.g dat or sed"),
                          prior_t_list=Param(str, "0.27 0.26 0.25 0.069 0.021 0.11 0.0061 0.0079", msg='String of numbers specifying prior type fracs MUST BE SAME LENGTH AS NUMBER OF SEDS'),
                          prior_zt_list=Param(str, "0.23 0.39 0.33 0.31 1.1 0.34 1.2 0.14", msg="string of numbers for redshift prior, MUST BE SAME LENGTH AS NUMBER OF SEDS"),
                          lambda_ref=Param(float, 4500., msg="referebce wavelength"),
                          train_refbandorder=Param(str, "DC2LSST_u DC2LSST_u_var DC2LSST_g DC2LSST_g_var DC2LSST_r DC2LSST_r_var DC2LSST_i DC2LSST_i_var DC2LSST_z DC2LSST_z_var DC2LSST_y DC2LSST_y_var redshift", msg="order of bands used in training"),
                          train_refband=Param(str, "DC2LSST_i", msg='reference band'),
                          train_fracfluxerr=Param(float, 1.e-4, msg="frac err to add to flux?"),
                          train_xvalidate=Param(bool, False, msg="perform cross validation flag"),
                          train_xvalbandorder=Param(str, "_ _ _ _ DC2LSST_r DC2LSST_r_var _ _ _ _ _ _", msg='band order for xval, unused bands indicated with _'),
                          gp_params_file=Param(str, "galaxies_gpparams.txt", msg='name of file to store gaussian process params fit by delightLearn'),
                          crossval_file=Param(str, "galaxies-gpCV.txt", msg='name of file to store crossvalidation parameters from delightLearn'),
                          target_refbandorder=Param(str, "DC2LSST_u DC2LSST_u_var DC2LSST_g DC2LSST_g_var DC2LSST_r DC2LSST_r_var DC2LSST_i DC2LSST_i_var DC2LSST_z DC2LSST_z_var DC2LSST_y DC2LSST_y_var redshift", msg='order of reference bands for target data'),
                          target_refband=Param(str, "DC2LSST_r", msg="the reference band for the taret data"),
                          target_fracfluxerr=Param(float, 1.e-4, msg="extra fractional error to add to target fluxes?"),
                          delightparamfile=Param(str, "parametersTest.cfg", msg="param file name"),
                          flag_filter_training=Param(bool, True, msg="?"),
                          snr_cut_training=Param(float, 5, msg="SNR training cut"),
                          flag_filter_validation=Param(bool, True, msg="?"),
                          snr_cut_validation=Param(float, 3, msg="validation SNR cut"),
                          dlght_inputdata=Param(str, "./examples/estimation/tmp/delight_indata", msg="input data directory for ascii data"),
                          zPriorSigma=Param(float, 0.2, msg="sigma for redshift prior"),
                          ellPriorSigma=Param(float, 0.5, msg="prior param"),
                          fluxLuminosityNorm=Param(float, 1.0, msg="luminosity norm factor"),
                          alpha_C=Param(float, 1.0e3, msg="prior param"),
                          V_C=Param(float, 0.1, msg="prior param"),
                          alpha_L=Param(float, 1.0e2, msg="prior param"),
                          V_L=Param(float, 0.1, msg="prior param"),
                          lineWidthSigma=Param(float, 20, msg="prior param"))

    outputs = []

    def __init__(self, args, comm=None):
        """ Constructor
        Do CatInformer specific initialization, then check on bands """
        CatInformer.__init__(self, args, comm=comm)
        # counter on the chunk validation dataset
        self.chunknum = 0
        self.delightparamfile = self.config['delightparamfile']

        np.random.seed(87)

    def inform(self, training_data):
        """Override the inform method because Delight doesn't have a model to return

        Parameters
        ----------
        input_data : `dict` or `TableHandle`
            dictionary of all input data, or a `TableHandle` providing access to it

        """
        self.set_data('input', training_data)
        self.run()
        self.finalize()

    def run(self):
        """Do all the annoying file IO stuff to ascii in current delight
           Then run delightApply to train the gauss. process
        """

        from delight.interfaces.rail.processFilters import processFilters
        from delight.interfaces.rail.processSEDs import processSEDs  # build a redshift -flux grid model
        from delight.interfaces.rail.makeConfigParam import makeConfigParam  # build the parameter file required by Delight
        from delight.interfaces.rail.convertDESCcat import convertDESCcatTrainData
        from delight.interfaces.rail.delightLearn import delightLearn

        try:
            if not os.path.exists(self.config['tempdir']):
                os.makedirs(self.config['tempdir'])  # pragma: no cover
        except OSError as e:  # pragma: no cover
            if e.errno != errno.EEXIST:
                msg = "error creating file " + self.config['tempdir']
                logger.error(msg)
                raise
        try:
            if not os.path.exists(self.config['tempdatadir']):
                os.makedirs(self.config['tempdatadir'])  # pragma: no cover
        except OSError as e:  # pragma: no cover
            if e.errno != errno.EEXIST:
                msg = "error creating file " + self.config['tempdatadir']
                logger.error(msg)
                raise

        basedelight_datapath = self.config['dlght_inputdata']
        # MAKE THE ASCII PARAM FILE THAT DELIGHT SCRIPTS READ
        paramfile_txt = makeConfigParam(basedelight_datapath, self.config)
        # save the config  parameter file that Delight needs
        with open(self.delightparamfile, 'w') as out:
            out.write(paramfile_txt)

        if not os.path.exists(self.config['sed_path']):  # pragma: no cover
            msg = " No Delight SED data in dir " + self.config['sed_path']
            logger.error(msg)
            sys.exit(-1)
        if not os.path.exists(self.config['bands_path']):  # pragma: no cover
            msg = " No Delight FILTER data in dir " + self.config['bands_path']
            logger.error(msg)
            sys.exit(-1)

        # Initialisation of Delight with 1) Filters 2) SED to get Flux-redshift model

        # Build LSST filter model
        processFilters(self.delightparamfile)

        # Build its own LSST-Flux-Redshift Model
        processSEDs(self.delightparamfile)

        # grab the training data
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data('input')

        convertDESCcatTrainData(self.delightparamfile,
                                training_data,
                                flag_filter=self.config['flag_filter_training'],
                                snr_cut=self.config['snr_cut_training'])

        # Learn with Gaussian processes
        delightLearn(self.delightparamfile)


class delightPZ(CatEstimator):
    """Run the delight scripts from the LSSTDESC fork of Delight
       Still has the ascii writeout stuff, so intermediate files are
       created that need to be cleaned up in the future
    """
    name = 'delightPZ'
    inputs = [('input', TableHandle)]
    config_options = CatEstimator.config_options.copy()
    config_options.update(dlght_redshiftMin=Param(float, 0.01, msg='min redshift'),
                          dlght_redshiftMax=Param(float, 3.01, msg='max redshift'),
                          dlght_redshiftNumBinsGPpred=Param(int, 301, msg='num bins'),
                          nzbins=Param(int, 301, msg="num z bins"),
                          dlght_redshiftBinSize=Param(float, 0.01, msg='???'),
                          dlght_redshiftDisBinSize=Param(float, 0.2, msg='bad, shouldnt be here'),
                          bands_names=Param(str, "DC2LSST_u DC2LSST_g DC2LSST_r DC2LSST_i DC2LSST_z DC2LSST_y", msg='string with list of Filter names'),
                          bands_path=Param(str, "./examples/estimation/data/FILTER", msg='string specifying path to filter directory'),
                          bands_fmt=Param(str, "res", msg="string giving the file extension of the filters, not including the '.'"),
                          bands_numcoefs=Param(int, 15, msg='integer specifying number of coefs in approximation of filter'),
                          bands_verbose=Param(bool, True, msg='verbose'),
                          bands_makeplots=Param(bool, False, msg='bool for whether to make approx band plots'),
                          bands_debug=Param(bool, True, msg='debug flag for filters'),
                          tempdir=Param(str, "./examples/estimation/tmp", msg='temp dir'),
                          tempdatadir=Param(str, "./examples/estimation/tmp/delight_data", msg='temp data dir'),
                          sed_path=Param(str, "./examples/estimation/data/SED", msg='path to SED dir'),
                          sed_name_list=Param(str, "El_B2004a Sbc_B2004a Scd_B2004a SB3_B2004a SB2_B2004a Im_B2004a ssp_25Myr_z008 ssp_5Myr_z008", msg='String with list of all SED names, with no file extension'),
                          sed_fmt=Param(str, "sed", msg="file extension of SED files (withough the '.', e.g dat or sed"),
                          prior_t_list=Param(str, "0.27 0.26 0.25 0.069 0.021 0.11 0.0061 0.0079", msg='String of numbers specifying prior type fracs MUST BE SAME LENGTH AS NUMBER OF SEDS'),
                          prior_zt_list=Param(str, "0.23 0.39 0.33 0.31 1.1 0.34 1.2 0.14", msg="string of numbers for redshift prior, MUST BE SAME LENGTH AS NUMBER OF SEDS"),
                          lambda_ref=Param(float, 4500., msg="referebce wavelength"),
                          train_refbandorder=Param(str, "DC2LSST_u DC2LSST_u_var DC2LSST_g DC2LSST_g_var DC2LSST_r DC2LSST_r_var DC2LSST_i DC2LSST_i_var DC2LSST_z DC2LSST_z_var DC2LSST_y DC2LSST_y_var redshift", msg="order of bands used in training"),
                          train_refband=Param(str, "DC2LSST_i", msg='reference band'),
                          train_fracfluxerr=Param(float, 1.e-4, msg="frac err to add to flux?"),
                          train_xvalidate=Param(bool, False, msg="perform cross validation flag"),
                          train_xvalbandorder=Param(str, "_ _ _ _ DC2LSST_r DC2LSST_r_var _ _ _ _ _ _", msg='band order for xval, unused bands indicated with _'),
                          gp_params_file=Param(str, "galaxies_gpparams.txt", msg='name of file to store gaussian process params fit by delightLearn'),
                          crossval_file=Param(str, "galaxies-gpCV.txt", msg='name of file to store crossvalidation parameters from delightLearn'),
                          target_refbandorder=Param(str, "DC2LSST_u DC2LSST_u_var DC2LSST_g DC2LSST_g_var DC2LSST_r DC2LSST_r_var DC2LSST_i DC2LSST_i_var DC2LSST_z DC2LSST_z_var DC2LSST_y DC2LSST_y_var redshift", msg='order of reference bands for target data'),
                          target_refband=Param(str, "DC2LSST_r", msg="the reference band for the taret data"),
                          target_fracfluxerr=Param(float, 1.e-4, msg="extra fractional error to add to target fluxes?"),
                          delightparamfile=Param(str, "parametersTest.cfg", msg="param file name"),
                          flag_filter_training=Param(bool, True, msg="?"),
                          snr_cut_training=Param(float, 5, msg="SNR training cut"),
                          flag_filter_validation=Param(bool, True, msg="?"),
                          snr_cut_validation=Param(float, 3, msg="validation SNR cut"),
                          dlght_inputdata=Param(str, "./examples/estimation/tmp/delight_indata", msg="input data directory for ascii data"),
                          zPriorSigma=Param(float, 0.2, msg="sigma for redshift prior"),
                          ellPriorSigma=Param(float, 0.5, msg="prior param"),
                          fluxLuminosityNorm=Param(float, 1.0, msg="luminosity norm factor"),
                          alpha_C=Param(float, 1.0e3, msg="prior param"),
                          V_C=Param(float, 0.1, msg="prior param"),
                          alpha_L=Param(float, 1.0e2, msg="prior param"),
                          V_L=Param(float, 0.1, msg="prior param"),
                          lineWidthSigma=Param(float, 20, msg="prior param"))

    def __init__(self, args, comm=None):
        """ Constructor:
        Do CatEstimator specific initialization """
        CatEstimator.__init__(self, args, comm=comm)
        self.delightparamfile = self.config['delightparamfile']
        self.chunknum = 0
        self.delightindata = self.config['dlght_inputdata']
        self.flag_filter_validation = self.config['flag_filter_validation']
        self.snr_cut_validation = self.config['snr_cut_validation']
        self.zgrid = np.arange(self.config['dlght_redshiftMin'], self.config['dlght_redshiftMax'], self.config['dlght_redshiftBinSize'])

    def open_model(self, **kwargs):
        """
        running TrainDelightPZ should create to ascii files whose locations
        are specified in the ascii param file, so we don't actually need to
        do anything here for now.
        """
        return


    def _process_chunk(self, start, end, data, first):

        from delight.interfaces.rail.makeConfigParam import makeConfigParam
        from delight.interfaces.rail.convertDESCcat import convertDESCcatChunk
        from delight.interfaces.rail.templateFitting import templateFitting
        from delight.interfaces.rail.delightApply import delightApply
        from delight.interfaces.rail.getDelightRedshiftEstimation import getDelightRedshiftEstimation

        print("\n\n\n Starting estimation...\n\n\n")
        self.chunknum += 1
        print(f"Process {self.rank} estimating PZ PDF for rows {start:,} - {end:,}")
        msg = f" ESTIMATE : chunk number {self.chunknum}"
        logger.info(msg)

        basedelight_datapath = self.delightindata

        msg = f" Delight input data file are in dir : {self.delightparamfile} "
        logger.debug(msg)

        msg = f"STANDARD MODE : process chunk {self.chunknum}"
        logger.info(msg)

        # Generate a new parameter file for delight this chunk
        paramfile_txt = makeConfigParam(basedelight_datapath, self.config, self.chunknum)

        # generate the config-parameter filename from chunk number
        delightparamfile = self.delightparamfile
        logger.debug(delightparamfile)
        dirn = os.path.dirname(delightparamfile)
        basn = os.path.basename(delightparamfile)
        basnsplit = basn.split(".")
        basnchunk = basnsplit[0] + "_" + str(self.chunknum) + "." + basnsplit[1]
        delightparamfilechunk = os.path.join(dirn, basnchunk)
        logger.debug("parameter file for delight :%s", delightparamfilechunk)

        # save the config parameter file for the data chunk that Delight needs
        with open(delightparamfilechunk, 'w') as out:
            out.write(paramfile_txt)

        # convert the chunk data into the required  flux-redshift validation file for delight
        indexes_sel = convertDESCcatChunk(delightparamfilechunk, data, self.chunknum,
                                          flag_filter_validation=self.flag_filter_validation,
                                          snr_cut_validation=self.snr_cut_validation)

        # template fitting for that chunk
        templateFitting(delightparamfilechunk)

        # estimation for that chunk
        delightApply(delightparamfilechunk)

        # allow for either format for now
        try:
            d = data['i_mag']
        except Exception:
            d = data['mag_i_lsst']

        numzs = len(d)

        zmode, pdfs = getDelightRedshiftEstimation(delightparamfilechunk,
                                                   self.chunknum, numzs, indexes_sel)
        zmode = np.round(zmode, 3)

        qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid,
                                                yvals=pdfs))
        qp_d.set_ancil(dict(zmode=zmode))
        self._do_chunk_output(qp_d, start, end, first)
