""" LSST Model for photometric errors """

from numbers import Number
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.stats import gaussian_kde
import pickle

from rail.creation.degradation import Degrader
import os

# folder containing data for the spec. success rate for the deep spec-z samples
success_rate_dir = os.path.join(
    os.path.dirname(__file__),  # location of this script
    "success_rate_data")

class SpecSelection(Degrader):
    """
    A toy degrader which only print out a given column

    Parameters
    ----------
    colname: string, the name of the spec survey
    """

    name = 'specselection'
    config_options = Degrader.config_options.copy()
    config_options.update(**{"N_tot": 10000})

    def __init__(self, args, comm=None):
        Degrader.__init__(self, args, comm=comm)
        # validate the settings
        self._validate_settings()


    def _validate_settings(self):
        """
        Validate all the settings.
        """

        # check that highSNR is boolean
        
        if isinstance(self.config["N_tot"], int) is not True:
            raise TypeError("Total number of selected sources must be an integer.")

    def downsampling_N_tot(self, data):
        """
        Method to randomly sample down the objects to a given
        number of data objects.
        """
        N_tot = self.config["N_tot"]
        
        idx_all = np.arange(len(data), dtype=np.int64)
        idx_preselect = idx_all[np.where(self.mask)[0]]
                
        # shuffle and select the first draw_n objects
        np.random.shuffle(idx_preselect)
        idx_keep = idx_preselect[:N_tot]
        # create a mask with only those entries enabled that have been selected
        mask = np.zeros_like(self.mask)
        mask[idx_keep] = True
        # update the internal state
        self.mask &= mask
        return
            
    def selection(self, data):
        return
        
    def run(self):
        """
        Run the toy
        """       
        
        # get the bands and bandNames present in the data
        data = self.get_data('input', allow_missing=True)        
        self.mask = np.product(~np.isnan(data.to_numpy()), axis=1)        

        #print("Applying the selection from "+selection+" survey...")
        self.selection(data)
        self.downsampling_N_tot(data)
        
        data_selected = data.iloc[np.where(self.mask==1)[0]]
        
        self.add_data('output', data_selected)


    def __repr__(self):  
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the selection."

        return printMsg

    
class SpecSelection_WiggleZ(SpecSelection):
    name = 'specselection_wigglez'
    def selection(self, data):
        print("Applying the selection from WiggleZ survey...")
        ###############################################################
        #   based on Drinkwater+10                                    #
        ###############################################################
        # colour masking
        colour_mask = (np.abs(data["g"]) < 99.0) & (np.abs(data["r"]) < 99.0)
        colour_mask &= (np.abs(data["r"]) < 99.0) & (np.abs(data["i"]) < 99.0)
        colour_mask &= (np.abs(data["u"]) < 99.0) & (np.abs(data["i"]) < 99.0)
        colour_mask &= (np.abs(data["g"]) < 99.0) & (np.abs(data["i"]) < 99.0)
        colour_mask &= (np.abs(data["r"]) < 99.0) & (np.abs(data["z"]) < 99.0)
        # photometric cuts
        # we cannot reproduce the FUV, NUV, S/N and position matching cuts
        include = (
            (data["r"] > 20.0) &
            (data["r"] < 22.5))
        exclude = (
            (data["g"] < 22.5) &
            (data["i"] < 21.5) &
            (data["r"]-data["i"] < (data["g"]-data["r"] - 0.1)) &
            (data["r"]-data["i"] < 0.4) &
            (data["g"]-data["r"] > 0.6) &
            (data["r"]-data["z"] < 0.7 * (data["g"]-data["r"])))
        mask = (include & ~exclude)
        # update the internal state
        self.mask *= mask
        return
        
    def __repr__(self):  
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the WiggleZ selection."

        return printMsg
    
    
class SpecSelection_GAMA(SpecSelection):
    name = 'specselection_gama'
    def selection(self, data):
        print("Applying the selection from GAMA survey...")
        self.mask *= (data["r"] < 17.7)
        return
        
    def __repr__(self):  
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the GAMA selection."

        return printMsg
    
    
class SpecSelection_BOSS(SpecSelection):
    name = 'specselection_boss'
    def selection(self, data):
        
        
        ###############################################################
        #   based on                                                  #
        #   http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php    #
        #   The selection has changed slightly compared to Dawson+13  #
        ###############################################################
        
        print("Applying the selection from BOSS survey...")
        mask = (np.abs(data["g"]) < 99.0) & (np.abs(data["r"]) < 99.0)
        mask &= (np.abs(data["r"]) < 99.0) & (np.abs(data["i"]) < 99.0)
        # cut quantities (unchanged)
        c_p = 0.7 * (data["g"]-data["r"]) + 1.2 * (data["r"]-data["i"] - 0.18)
        c_r = (data["r"]-data["i"]) - (data["g"]-data["r"]) / 4.0 - 0.18
        d_r = (data["r"]-data["i"]) - (data["g"]-data["r"]) / 8.0
        # defining the LOWZ sample
        # we cannot apply the r_psf - r_cmod cut
        low_z = (
            (data["r"] > 16.0) &
            (data["r"] < 20.0) &  # 19.6
            (np.abs(c_r) < 0.2) &
            (data["r"] < 13.35 + c_p / 0.3))  # 13.5, 0.3
        # defining the CMASS sample
        # we cannot apply the i_fib2, i_psf - i_mod and z_psf - z_mod cuts
        cmass = (
            (data["i"] > 17.5) &
            (data["i"] < 20.1) &  # 19.9
            (d_r > 0.55) &
            (data["i"] < 19.98 + 1.6 * (d_r - 0.7)) &  # 19.86, 1.6, 0.8
            ((data["r"]-data["i"]) < 2.0))
        # NOTE: we ignore the CMASS sparse sample
        self.mask *= (low_z | cmass)
        return
        
    def __repr__(self):  
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the BOSS selection."

        return printMsg
    
    
class SpecSelection_DEEP2(SpecSelection):
    
    name = 'specselection_deep2'
    
    def photometryCut(self, data):
        ###############################################################
        #   based on Newman+13                                        #
        ###############################################################
        # this modified selection gives the best match to the data n(z) with
        # its cut at z~0.75 and the B-R/R-I distribution (Newman+13, Fig. 12)
        # NOTE: We cannot apply the surface brightness cut and do not apply the
        #       Gaussian weighted sampling near the original colour cuts.
        mask = (
            (data["r"] > 18.5) &
            (data["r"] < 24.0) & (  # 24.1
                (data["g"]-data["r"] < 2.0 * (data["r"]-data["i"]) - 0.4) |  # 2.45, 0.2976
                (data["r"]-data["i"] > 1.1) |
                (data["g"]-data["r"] < 0.2)))  # 0.5
        # update the internal state
        self.mask &= mask
        return 

    def speczSuccess(self, data):
        # Spec-z success rate as function of r_AB for Q>=3 read of Figure 13 in
        # Newman+13 for DEEP2 fields 2-4. Values are binned in steps of 0.2 mag
        # with the first and last bin centered on 19 and 24.
        success_R_bins = np.arange(18.9, 24.1 + 0.01, 0.2)
        success_R_centers = (success_R_bins[1:] + success_R_bins[:-1]) / 2.0
        # paper has given 1 - [sucess rate] in the histogram
        
        import os
        print(os.system('pwd'))
        
        success_R_rate = np.loadtxt(os.path.join(success_rate_dir,"DEEP2_success.txt"))
        # interpolate the success rate as probability of being selected with
        # the probability at R > 24.1 being 0
        p_success_R = interp1d(
            success_R_centers, success_R_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_R_rate[0], 0.0))
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(data))
        mask = random_draw < p_success_R(data["r"])
        # update the internal state
        self.mask &= mask
        return 
        
    def selection(self, data):
        self.photometryCut(data)
        self.speczSuccess(data)
        return
        
        
    def __repr__(self):  
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the DEEP2 selection."

        return printMsg
    
    
class SpecSelection_VVDSf02(SpecSelection):
    
    name = 'specselection_VVDSf02'
    
    def photometryCut(self, data):
        ###############################################################
        #   based on LeFèvre+05                                       #
        ###############################################################
        mask = (data["i"] > 18.5) & (data["i"] < 24.0)  # 17.5, 24.0
        # NOTE: The oversight of 1.0 magnitudes on the bright end misses 0.2 %
        #       of galaxies.
        # update the internal state
        self.mask &= mask
        return 

    def speczSuccess(self, data):
        # NOTE: We use a redshift-based and I-band based success rate
        #       independently here since we do not know their correlation,
        #       which makes the success rate worse than in reality.
        # Spec-z success rate as function of i_AB read of Figure 16 in
        # LeFevre+05 for the VVDS 2h field. Values are binned in steps of
        # 0.5 mag with the first starting at 17 and the last bin ending at 24.
        success_I_bins = np.arange(17.0, 24.0 + 0.01, 0.5)
        success_I_centers = (success_I_bins[1:] + success_I_bins[:-1]) / 2.0
        success_I_rate = np.loadtxt(os.path.join(
                success_rate_dir, "VVDSf02_I_success.txt"))
        # interpolate the success rate as probability of being selected with
        # the probability at I > 24 being 0
        p_success_I = interp1d(
            success_I_centers, success_I_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_I_rate[0], 0.0))
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(data))
        mask = random_draw < p_success_I(data["i"])
        # Spec-z success rate as function of redshift read of Figure 13a/b in
        # LeFevre+13 for VVDS deep sample. The listing is split by i_AB into
        # ranges (17.5; 22.5] and (22.5; 24.0].
        # NOTE: at z > 1.75 there are only lower limits (due to a lack of
        # spec-z?), thus the success rate is extrapolated as 1.0 at z > 1.75
        success_z_bright_centers, success_z_bright_rate = np.loadtxt(
            os.path.join(success_rate_dir, "VVDSf02_z_bright_success.txt")).T
        success_z_deep_centers, success_z_deep_rate = np.loadtxt(
            os.path.join(success_rate_dir, "VVDSf02_z_deep_success.txt")).T
        # interpolate the success rates as probability of being selected with
        # the probability in the bright bin at z > 1.75 being 1.0 and the deep
        # bin at z > 4.0 being 0.0
        p_success_z_bright = interp1d(
            success_z_bright_centers, success_z_bright_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_z_bright_rate[0], 1.0))
        p_success_z_deep = interp1d(
            success_z_deep_centers, success_z_deep_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_z_deep_rate[0], 0.0))
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(data))
        iterator = zip(
            [data["i"] <= 22.5, data["i"] > 22.5],
            [p_success_z_bright, p_success_z_deep])
        for m, p_success_z in iterator:
            mask[m] &= random_draw[m] < p_success_z(data["redshift"][m])
        # update the internal state
        self.mask &= mask
        return 
        
    def selection(self, data):
        self.photometryCut(data)
        self.speczSuccess(data)
        return
        
        
    def __repr__(self):  
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the VVDSf02 selection."

        return printMsg
    
    
class SpecSelection_zCOSMOS(SpecSelection):
    
    name = 'specselection_zCOSMOS'
    
    def photometryCut(self, data):
        ###############################################################
        #   based on Lilly+09                                         #
        ###############################################################
        mask = (data["i"] > 15.0) & (data["i"] < 22.5)  # 15.0, 22.5
        # NOTE: This only includes zCOSMOS bright.
        # update the internal state
        self.mask &= mask
        return 

    def speczSuccess(self, data):
        # Spec-z success rate as function of redshift (x) and I_AB (y) read of
        # Figure 3 in Lilly+09 for zCOSMOS bright sample. Do a spline
        # interpolation of the 2D data and save it as pickle on the disk for
        # faster reloads
        pickle_file = os.path.join(success_rate_dir, "zCOSMOS.cache")
        if not os.path.exists(pickle_file):
            x = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_z_sampling.txt"))
            y = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_I_sampling.txt"))
            rates = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_success.txt"))
            p_success_zI = interp2d(x, y, rates, copy=True, kind="linear")
            with open(pickle_file, "wb") as f:
                pickle.dump(p_success_zI, f)
        else:
            with open(pickle_file, "rb") as f:
                p_success_zI = pickle.load(f)
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(data))
        object_rates = np.empty_like(random_draw)
        for i, (z, I_AB) in enumerate(zip(data["redshift"], data["i"])):
            # this must be in a loop since interp2d will create a grid from the
            # input redshifts and magnitudes instead of evaluating pairs of
            # values
            object_rates[i] = p_success_zI(z, I_AB)
        mask = random_draw < object_rates
        # update the internal state
        self.mask &= mask
        return 
        
    def selection(self, data):
        self.photometryCut(data)
        self.speczSuccess(data)
        return
        
        
    def __repr__(self):  
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the zCOSMOS selection."

        return printMsg