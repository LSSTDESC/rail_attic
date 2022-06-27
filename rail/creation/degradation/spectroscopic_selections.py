""" Applying selection functions to catalog """

import numpy as np
from scipy.interpolate import interp1d


from rail.creation.degradation import Degrader
import os


class SpecSelection(Degrader):
    """
    The super class of spectroscopic selections.

    Parameters
    ----------
    colname: string, the name of the spec survey
    """

    name = 'specselection'
    config_options = Degrader.config_options.copy()
    config_options.update(**{"N_tot": 10000})
    config_options.update(**{"downsample": True})
    config_options.update(**{"success_rate_dir":
                             os.path.join(
                                 os.path.dirname(__file__),
                                 "success_rate_data")})

    def __init__(self, args, comm=None):
        Degrader.__init__(self, args, comm=comm)
        # validate the settings
        self._validate_settings()
        self.mask = None

    def _validate_settings(self):
        """
        Validate all the settings.
        """

        # check that highSNR is boolean
        if isinstance(self.config["N_tot"], int) is not True:
            raise TypeError("Total number of selected sources must be an "
                            "integer.")
        if os.path.exists(self.config["success_rate_dir"]) is not True:
            raise ValueError("Success rate path: "
                             + self.config["success_rate_dir"]
                             + " does not exist!")

    def validate_colnames(self, data):
        """
        Validate the column names of data table to make sure they have
        the format of mag_{band}_lsst, where band is in {ugrizy}
        """
        standard_colnames = [f'mag_{band}_lsst' for band in 'ugrizy']
        check = all(item in data.columns for item in standard_colnames)
        if check is not True:
            raise ValueError("Column names in the data is not standardized." +
                             "The standard column names should be " +
                             str(standard_colnames)+". \n" +
                             "You can make a ColumnMapper stage to change " +
                             "column names.")

    def selection(self, data):
        """
        Selection functions
        """

    def downsampling_N_tot(self):
        """
        Method to randomly sample down the objects to a given
        number of data objects.
        """
        N_tot = self.config["N_tot"]
        N_selected = np.where(self.mask)[0].size
        if N_tot > N_selected:
            print("Warning: N_tot is greater than the size of spec-selected " +
                  "sample ("+str(N_selected)+"). The spec-selected sample " +
                  "is returned.")
            return
        else:
            idx_selected = np.where(self.mask)[0]
            idx_keep = np.random.choice(idx_selected, replace=False,
                                        size=N_tot)
            # create a mask with only those entries enabled that have been
            # selected
            mask = np.zeros_like(self.mask)
            mask[idx_keep] = True
            # update the internal state
            self.mask &= mask

    def run(self):
        """
        Run the selection
        """

        # get the bands and bandNames present in the data
        data = self.get_data('input', allow_missing=True)
        self.validate_colnames(data)
        self.mask = np.product(~np.isnan(data.to_numpy()), axis=1)

        self.selection(data)
        if self.config["downsample"] is True:
            self.downsampling_N_tot()

        data_selected = data.iloc[np.where(self.mask == 1)[0]]

        self.add_data('output', data_selected)

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the selection."

        return printMsg


class SpecSelection_WiggleZ(SpecSelection):
    """
    The class of spectroscopic selections with WiggleZ.
    """

    name = 'specselection_wigglez'

    def selection(self, data):
        """
        WiggleZ selection function based on Drinkwater+10
        """
        print("Applying the selection from WiggleZ survey...")
        # colour masking
        colour_mask = (np.abs(data["mag_g_lsst"]) < 99.0) & \
            (np.abs(data["mag_r_lsst"]) < 99.0)
        colour_mask &= (np.abs(data["mag_r_lsst"]) < 99.0) & \
            (np.abs(data["mag_i_lsst"]) < 99.0)
        colour_mask &= (np.abs(data["mag_u_lsst"]) < 99.0) & \
            (np.abs(data["mag_i_lsst"]) < 99.0)
        colour_mask &= (np.abs(data["mag_g_lsst"]) < 99.0) & \
            (np.abs(data["mag_i_lsst"]) < 99.0)
        colour_mask &= (np.abs(data["mag_r_lsst"]) < 99.0) & \
            (np.abs(data["mag_z_lsst"]) < 99.0)
        # photometric cuts
        # we cannot reproduce the FUV, NUV, S/N and position matching cuts
        include = (
            (data["mag_r_lsst"] > 20.0) &
            (data["mag_r_lsst"] < 22.5))
        exclude = (
            (data["mag_g_lsst"] < 22.5) &
            (data["mag_i_lsst"] < 21.5) &
            (data["mag_r_lsst"]-data["mag_i_lsst"] <
             (data["mag_g_lsst"]-data["mag_r_lsst"] - 0.1)) &
            (data["mag_r_lsst"]-data["mag_i_lsst"] < 0.4) &
            (data["mag_g_lsst"]-data["mag_r_lsst"] > 0.6) &
            (data["mag_r_lsst"]-data["mag_z_lsst"] < 0.7 *
             (data["mag_g_lsst"]-data["mag_r_lsst"])))
        mask = (include & ~exclude)
        # update the internal state
        self.mask *= mask

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the WiggleZ selection."

        return printMsg


class SpecSelection_GAMA(SpecSelection):
    """
    The class of spectroscopic selections with GAMA.
    """

    name = 'specselection_gama'

    def selection(self, data):
        """
        GAMA selection function
        """

        print("Applying the selection from GAMA survey...")
        self.mask *= (data["mag_r_lsst"] < 17.7)

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the GAMA selection."

        return printMsg


class SpecSelection_BOSS(SpecSelection):
    """
    The class of spectroscopic selections with BOSS.
    """

    name = 'specselection_boss'

    def selection(self, data):
        """
        BOSS selection function based on
        http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
        The selection has changed slightly compared to Dawson+13
        """

        print("Applying the selection from BOSS survey...")
        mask = (np.abs(data["mag_g_lsst"]) < 99.0) & \
            (np.abs(data["mag_r_lsst"]) < 99.0)
        mask &= (np.abs(data["mag_r_lsst"]) < 99.0) & \
            (np.abs(data["mag_i_lsst"]) < 99.0)
        # cut quantities (unchanged)
        c_p = 0.7 * (data["mag_g_lsst"]-data["mag_r_lsst"]) + 1.2 * \
            (data["mag_r_lsst"]-data["mag_i_lsst"] - 0.18)
        c_r = (data["mag_r_lsst"]-data["mag_i_lsst"]) - \
            (data["mag_g_lsst"]-data["mag_r_lsst"]) / 4.0 - 0.18
        d_r = (data["mag_r_lsst"]-data["mag_i_lsst"]) - \
            (data["mag_g_lsst"]-data["mag_r_lsst"]) / 8.0
        # defining the LOWZ sample
        # we cannot apply the r_psf - r_cmod cut
        low_z = (
            (data["mag_r_lsst"] > 16.0) &
            (data["mag_r_lsst"] < 20.0) &  # 19.6
            (np.abs(c_r) < 0.2) &
            (data["mag_r_lsst"] < 13.35 + c_p / 0.3))  # 13.5, 0.3
        # defining the CMASS sample
        # we cannot apply the i_fib2, i_psf - i_mod and z_psf - z_mod cuts
        cmass = (
            (data["mag_i_lsst"] > 17.5) &
            (data["mag_i_lsst"] < 20.1) &  # 19.9
            (d_r > 0.55) &
            (data["mag_i_lsst"] < 19.98 + 1.6 * (d_r - 0.7)) &  # 19.86, 1.6, 0.8
            ((data["mag_r_lsst"]-data["mag_i_lsst"]) < 2.0))
        # NOTE: we ignore the CMASS sparse sample
        self.mask *= (low_z | cmass)

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the BOSS selection."

        return printMsg


class SpecSelection_DEEP2(SpecSelection):
    """
    The class of spectroscopic selections with DEEP2.
    """

    name = 'specselection_deep2'

    def photometryCut(self, data):
        """
        Applying DEEP2 photometric cut based on Newman+13.
        This modified selection gives the best match to the data n(z) with
        its cut at z~0.75 and the B-R/R-I distribution (Newman+13, Fig. 12)
        NOTE: We cannot apply the surface brightness cut and do not apply the
              Gaussian weighted sampling near the original colour cuts.

        """
        mask = (
            (data["mag_r_lsst"] > 18.5) &
            (data["mag_r_lsst"] < 24.0) & (  # 24.1
                (data["mag_g_lsst"]-data["mag_r_lsst"] < 2.0 * \
                 (data["mag_r_lsst"]-data["mag_i_lsst"]) - 0.4) |
                # 2.45, 0.2976
                (data["mag_r_lsst"]-data["mag_i_lsst"] > 1.1) |
                (data["mag_g_lsst"]-data["mag_r_lsst"] < 0.2)))  # 0.5
        # update the internal state
        self.mask &= mask

    def speczSuccess(self, data):
        """
        Spec-z success rate as function of r_AB for Q>=3 read of Figure 13 in
        Newman+13 for DEEP2 fields 2-4. Values are binned in steps of 0.2 mag
        with the first and last bin centered on 19 and 24.
        """
        success_R_bins = np.arange(18.9, 24.1 + 0.01, 0.2)
        success_R_centers = (success_R_bins[1:] + success_R_bins[:-1]) / 2.0
        # paper has given 1 - [sucess rate] in the histogram
        success_rate_dir = self.config["success_rate_dir"]
        success_R_rate = np.loadtxt(os.path.join(success_rate_dir,
                                                 "DEEP2_success.txt"))
        # interpolate the success rate as probability of being selected with
        # the probability at R > 24.1 being 0
        p_success_R = interp1d(
            success_R_centers, success_R_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_R_rate[0], 0.0))
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(data))
        mask = random_draw < p_success_R(data["mag_r_lsst"])
        # update the internal state
        self.mask &= mask

    def selection(self, data):
        """
        DEEP2 selection function
        """
        self.photometryCut(data)
        self.speczSuccess(data)

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the DEEP2 selection."

        return printMsg


class SpecSelection_VVDSf02(SpecSelection):
    """
    The class of spectroscopic selections with VVDSf02.
    """

    name = 'specselection_VVDSf02'

    def photometryCut(self, data):
        """
        Photometric cut of VVDS 2h-field based on LeFèvre+05.
        NOTE: The oversight of 1.0 magnitudes on the bright end misses 0.2 %
               of galaxies.
        update the internal state
        """
        mask = (data["mag_i_lsst"] > 18.5) & (data["mag_i_lsst"] < 24.0)
        # 17.5, 24.0
        self.mask &= mask

    def speczSuccess(self, data):
        """
        Success rate of VVDS 2h-field
        NOTE: We use a redshift-based and I-band based success rate
               independently here since we do not know their correlation,
               which makes the success rate worse than in reality.
        Spec-z success rate as function of i_AB read of Figure 16 in
        LeFevre+05 for the VVDS 2h field. Values are binned in steps of
        0.5 mag with the first starting at 17 and the last bin ending at 24.
        """
        success_I_bins = np.arange(17.0, 24.0 + 0.01, 0.5)
        success_I_centers = (success_I_bins[1:] + success_I_bins[:-1]) / 2.0
        success_rate_dir = self.config["success_rate_dir"]
        success_I_rate = np.loadtxt(os.path.join(
                success_rate_dir, "VVDSf02_I_success.txt"))
        # interpolate the success rate as probability of being selected with
        # the probability at I > 24 being 0
        p_success_I = interp1d(
            success_I_centers, success_I_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_I_rate[0], 0.0))
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(data))
        mask = random_draw < p_success_I(data["mag_i_lsst"])
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
            [data["mag_i_lsst"] <= 22.5, data["mag_i_lsst"] > 22.5],
            [p_success_z_bright, p_success_z_deep])
        for m, p_success_z in iterator:
            mask[m] &= random_draw[m] < p_success_z(data["redshift"][m])
        # update the internal state
        self.mask &= mask

    def selection(self, data):
        self.photometryCut(data)
        self.speczSuccess(data)

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the VVDSf02 selection."

        return printMsg


class SpecSelection_zCOSMOS(SpecSelection):
    """
    The class of spectroscopic selections with zCOSMOS
    """

    name = 'specselection_zCOSMOS'

    def photometryCut(self, data):
        """
        Photometry cut for zCOSMOS based on Lilly+09.
        NOTE: This only includes zCOSMOS bright.
        update the internal state
        """
        mask = (data["mag_i_lsst"] > 15.0) & (data["mag_i_lsst"] < 22.5)
        # 15.0, 22.5
        self.mask &= mask

    def speczSuccess(self, data):
        """
        Spec-z success rate as function of redshift (x) and I_AB (y) read of
        Figure 3 in Lilly+09 for zCOSMOS bright sample. Do a spline
        interpolation of the 2D data and save it as pickle on the disk for
        faster reloads
        """
        success_rate_dir = self.config["success_rate_dir"]
        x = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_z_sampling.txt"))
        y = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_I_sampling.txt"))

        pixels_y = np.searchsorted(y, data["mag_i_lsst"])
        pixels_x = np.searchsorted(x, data["redshift"])

        rates = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_success.txt"))
        ratio_list = []
        for i, py in enumerate(pixels_y):
            if (py >= rates.shape[0]) or \
               (pixels_x[i] >= rates.shape[1]):
                rate = 0
            else:
                rate = rates[pixels_y[i]][pixels_x[i]]
            ratio_list.append(rate)

        ratio_list = np.array(ratio_list)
        randoms = np.random.uniform(size=data["mag_i_lsst"].size)
        mask = (randoms <= ratio_list)
        self.mask &= mask

    def selection(self, data):
        self.photometryCut(data)
        self.speczSuccess(data)

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the zCOSMOS selection."

        return printMsg


class SpecSelection_HSC(SpecSelection):
    """
    The class of spectroscopic selections with HSC
    """

    name = 'specselection_HSC'

    def photometryCut(self, data):
        """
        HSC galaxies were binned in color magnitude space with i-band mag from -2 to 6 and g-z color from 13 to 26.
        """
        mask = (data["mag_i_lsst"] > 13.0) & (data["mag_i_lsst"] < 26.)
        self.mask &= mask
        gz = data["mag_g_lsst"] - data["mag_z_lsst"]
        mask = (gz > -2.) & (gz < 6.)
        self.mask &= mask

    def speczSuccess(self, data):
        """
        HSC galaxies were binned in color magnitude space with i-band mag from -2 to 6 and g-z color from 13 to 26
        200 bins in each direction. The ratio of of galaxies with spectroscopic redshifts (training galaxies) to
        galaxies with only photometry in HSC wide field (application galaxies) was computed for each pixel. We divide
        the data into the same pixels and randomly select galaxies into the training sample based on the HSC ratios
        """
        success_rate_dir = self.config["success_rate_dir"]
        x_edge = np.loadtxt(os.path.join(
                success_rate_dir, "hsc_i_binedge.txt"))
        y_edge = np.loadtxt(os.path.join(
                success_rate_dir, "hsc_gz_binedge.txt"))

        rates = np.loadtxt(os.path.join(
                success_rate_dir, "hsc_success.txt"))

        pixels_y = np.searchsorted(y_edge, data["mag_g_lsst"]-data["mag_z_lsst"])
        pixels_x = np.searchsorted(x_edge, data["mag_i_lsst"])

        pixels_y = pixels_y - 1
        pixels_x = pixels_x - 1

        ratio_list = []
        for i, py in enumerate(pixels_y):
            if (py >= rates.shape[0]) or\
               (pixels_x[i] >= rates.shape[1]):
                rate = 0
            else:
                rate = rates[pixels_y[i]][pixels_x[i]]
            ratio_list.append(rate)

        ratio_list = np.array(ratio_list)
        randoms = np.random.uniform(size=data["mag_i_lsst"].size)
        mask = (randoms <= ratio_list)
        self.mask &= mask

    def selection(self, data):
        self.photometryCut(data)
        self.speczSuccess(data)

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """
        # start message
        printMsg = "Applying the HSC selection."

        return printMsg
