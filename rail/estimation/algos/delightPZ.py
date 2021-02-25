"""
Example code that just spits out random numbers between 0 and 3
for z_mode, and Gaussian centered at z_mode with width
random_width*(1+zmode).
"""

import numpy as np
from scipy.stats import norm
from rail.estimation.estimator import Estimator as BaseEstimation

import os

import coloredlogs
import logging
from pkg_resources import resource_filename

from interfaces.rail.processFilters import processFilters  # interface added into delight in branch rail
from interfaces.rail.makeConfigParam import makeConfigParam  # build the parameter file required by Delight

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
            os.makedirs(self.tempdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        try:
            os.makedirs(self.tempdatadir)
        except OSError as e:
            if e.errno != errno.EEXIST:
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

        # Build LSST filter model
        processFilters(self.delightparamfile)

    #############################################################################################################
    #
    # ESTIMATE
    #
    ############################################################################################################


    def estimate(self, test_data):
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
