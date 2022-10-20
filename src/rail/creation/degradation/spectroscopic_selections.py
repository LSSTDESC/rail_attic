""" Applying selection functions to catalog """

import os

import numpy as np
from ceci.config import StageParameter as Param
from rail.creation.degrader import Degrader
from scipy.interpolate import interp1d
from rail.core.utils import RAILDIR


class SpecSelection(Degrader):
    """
    The super class of spectroscopic selections.

    Parameters
    ----------
    N_tot: integer
        total number of down-sampled, spec-selected galaxies.
        If N_tot is greater than the number of spec-sepected galaxies, then
        it will be ignored.
    nondetect_val: value to be removed for non detects
    downsample: bool
        If True, then downsample the pre-selected galaxies
        to N_tot galaxies.
    success_rate_dir: string, the path to the success rate files.
    percentile_cut: If using color-based redshift cut, percentile in redshifts above which redshifts will be cut from the sample. Default is 100 (no cut)
    colnames: a dictionary that includes necessary columns
                         (magnitudes, colors and redshift) for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
                         for example, gr standing for g-r; for redshift, the key is 'redshift'.
    random_seed: random seed for reproducibility.

    """

    name = "specselection"
    config_options = Degrader.config_options.copy()
    config_options.update(
        N_tot=Param(int, 10000, msg="Number of selected sources"),
        nondetect_val=Param(float, 99.0, msg="value to be removed for non detects"),
        downsample=Param(
            bool,
            True,
            msg="If true, downsample the selected sources into a total number of N_tot",
        ),
        success_rate_dir=Param(
            str,
            os.path.join(RAILDIR,
                "rail/examples/creation/data/success_rate_data",
            ),
            msg="The path to the directory containing success rate files.",
        ),
        percentile_cut=Param(int, 100, msg="cut redshifts above this percentile"),
        colnames=Param(
            dict,
            {
                **{band: "mag_" + band + "_lsst" for band in "ugrizy"},
                **{"redshift": "redshift"},
            },
            msg="a dictionary that includes necessary columns\
                         (magnitudes, colors and redshift) for selection. For magnitudes, the keys are ugrizy; for colors, the keys are, \
                         for example, gr standing for g-r; for redshift, the key is 'redshift'",
        ),
        random_seed=Param(int, 42, msg="random seed for reproducibility"),
    )

    def __init__(self, args, comm=None):
        Degrader.__init__(self, args, comm=comm)
        # validate the settings
        self._validate_settings()
        self.mask = None
        self.rng = None

    def _validate_settings(self):
        """
        Validate all the settings.
        """

        if self.config.N_tot < 0:
            raise ValueError(
                "Total number of selected sources must be a " "positive integer."
            )
        if os.path.exists(self.config.success_rate_dir) is not True:
            raise ValueError(
                "Success rate path: "
                + self.config.success_rate_dir
                + " does not exist!"
            )

    def validate_colnames(self, data):
        """
        Validate the column names of data table to make sure they have necessary information
        for each selection.
        colnames: a list of column names.
        """
        colnames = self.config.colnames.values()
        check = all(item in data.columns for item in colnames)
        if check is not True:
            raise ValueError(
                "Columns in the data are not enough for the selection."
                + "The data should contain "
                + str(list(colnames))
                + ". \n"
            )

    def selection(self, data):
        """
        Selection functions. This should be overwritten by the subclasses
        corresponding to different spec selections.
        """

    def invalid_cut(self, data):
        """
        This function removes entries in the data that have invalid magnitude values
        (nondetect_val or NaN)
        """
        nondetect_val = self.config.nondetect_val
        for band in "ugrizy":
            if band not in self.config.colnames.keys():
                continue
            colname = self.config.colnames[band]
            self.mask &= (np.abs(data[colname]) < nondetect_val) & (
                ~np.isnan(data[colname])
            )

    def downsampling_N_tot(self):
        """
        Method to randomly sample down the objects to a given
        number of data objects.
        """
        N_tot = self.config.N_tot
        N_selected = np.count_nonzero(self.mask)
        if N_tot > N_selected:
            print(
                "Warning: N_tot is greater than the size of spec-selected "
                + "sample ("
                + str(N_selected)
                + "). The spec-selected sample "
                + "is returned."
            )
            return
        else:
            idx_selected = np.where(self.mask)[0]
            idx_keep = self.rng.choice(idx_selected, replace=False, size=N_tot)
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
        self.rng = np.random.default_rng(seed=self.config.seed)
        # get the bands and bandNames present in the data
        data = self.get_data("input", allow_missing=True)
        self.validate_colnames(data)
        self.mask = np.product(~np.isnan(data.to_numpy()), axis=1)
        self.invalid_cut(data)
        self.selection(data)
        if self.config.downsample is True:
            self.downsampling_N_tot()

        data_selected = data.iloc[np.where(self.mask == 1)[0]]

        self.add_data("output", data_selected)

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """


class SpecSelection_GAMA(SpecSelection):
    """
    The class of spectroscopic selections with GAMA.
    The GAMA survey covers an area of 286 deg^2, with ~238000 objects
    The necessary column is r band
    """

    name = "specselection_gama"

    def selection(self, data):
        """
        GAMA selection function
        """
        print("Applying the selection from GAMA survey...")
        self.mask *= data[self.config.colnames["r"]] < 19.87

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
    BOSS selection function is based on
    http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
    The selection has changed slightly compared to Dawson+13
    BOSS covers an area of 9100 deg^2 with 893,319 galaxies.
    For BOSS selection, the data should at least include gri bands.
    """

    name = "specselection_boss"

    def selection(self, data):
        """
        The BOSS selection function.
        """

        print("Applying the selection from BOSS survey...")
        # cut quantities (unchanged)
        c_p = 0.7 * (
            data[self.config.colnames["g"]] - data[self.config.colnames["r"]]
        ) + 1.2 * (
            data[self.config.colnames["r"]] - data[self.config.colnames["i"]] - 0.18
        )
        c_r = (
            (data[self.config.colnames["r"]] - data[self.config.colnames["i"]])
            - (data[self.config.colnames["g"]] - data[self.config.colnames["r"]]) / 4.0
            - 0.18
        )
        d_r = (data[self.config.colnames["r"]] - data[self.config.colnames["i"]]) - (
            data[self.config.colnames["g"]] - data[self.config.colnames["r"]]
        ) / 8.0
        # defining the LOWZ sample
        # we cannot apply the r_psf - r_cmod cut
        low_z = (
            (data[self.config.colnames["r"]] > 16.0)
            & (data[self.config.colnames["r"]] < 20.0)
            & (np.abs(c_r) < 0.2)  # 19.6
            & (data[self.config.colnames["r"]] < 13.35 + c_p / 0.3)
        )  # 13.5, 0.3
        # defining the CMASS sample
        # we cannot apply the i_fib2, i_psf - i_mod and z_psf - z_mod cuts
        cmass = (
            (data[self.config.colnames["i"]] > 17.5)
            & (data[self.config.colnames["i"]] < 20.1)
            & (d_r > 0.55)  # 19.9
            & (data[self.config.colnames["i"]] < 19.98 + 1.6 * (d_r - 0.7))
            & (  # 19.86, 1.6, 0.8
                (data[self.config.colnames["r"]] - data[self.config.colnames["i"]])
                < 2.0
            )
        )
        # NOTE: we ignore the CMASS sparse sample
        self.mask *= low_z | cmass

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
    DEEP2 has a sky coverage of 2.8 deg^2 with ~53000 spectra.
    For DEEP2, one needs R band magnitude, B-R/R-I colors which are not available for the time being.
    So we use LSST gri bands now. When the conversion degrader is ready, this subclass will be updated
    accordingly.
    """

    name = "specselection_deep2"

    def photometryCut(self, data):
        """
        Applying DEEP2 photometric cut based on Newman+13.
        This modified selection gives the best match to the data n(z) with
        its cut at z~0.75 and the B-R/R-I distribution (Newman+13, Fig. 12)

        Notes
        -----
        We cannot apply the surface brightness cut and do not apply the
        Gaussian weighted sampling near the original colour cuts.
        """
        mask = (
            (data[self.config.colnames["r"]] > 18.5)
            & (data[self.config.colnames["r"]] < 24.1)
            & (  # 24.1
                (
                    data[self.config.colnames["g"]] - data[self.config.colnames["r"]]
                    < 2.45
                    * (
                        data[self.config.colnames["r"]]
                        - data[self.config.colnames["i"]]
                    )
                    - 0.2976
                )
                |
                # 2.45, 0.2976
                (
                    data[self.config.colnames["r"]] - data[self.config.colnames["i"]]
                    > 1.1
                )
                | (
                    data[self.config.colnames["g"]] - data[self.config.colnames["r"]]
                    < 0.5
                )
            )
        )  # 0.5
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
        success_rate_dir = self.config.success_rate_dir
        success_R_rate = np.loadtxt(os.path.join(success_rate_dir, "DEEP2_success.txt"))
        # interpolate the success rate as probability of being selected with
        # the probability at R > 24.1 being 0
        p_success_R = interp1d(
            success_R_centers,
            success_R_rate,
            kind="quadratic",
            bounds_error=False,
            fill_value=(success_R_rate[0], 0.0),
        )
        # Randomly sample objects according to their success rate
        random_draw = self.rng.random(len(data))
        mask = random_draw < p_success_R(data[self.config.colnames["r"]])
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
    It covers an area of 0.5 deg^2 with ~10000 sources.
    Necessary columns are i band magnitude and redshift.
    """

    name = "specselection_VVDSf02"

    def photometryCut(self, data):
        """
        Photometric cut of VVDS 2h-field based on LeFèvre+05.

        Notes
        -----
        The oversight of 1.0 magnitudes on the bright end misses 0.2% of galaxies.
        """
        mask = (data[self.config.colnames["i"]] > 17.5) & (
            data[self.config.colnames["i"]] < 24.0
        )
        # 17.5, 24.0
        self.mask &= mask

    def speczSuccess(self, data):
        """
        Success rate of VVDS 2h-field

        Notes
        -----
        We use a redshift-based and I-band based success rate
        independently here since we do not know their correlation,
        which makes the success rate worse than in reality.

        Spec-z success rate as function of i_AB read of Figure 16 in
        LeFevre+05 for the VVDS 2h field. Values are binned in steps of
        0.5 mag with the first starting at 17 and the last bin ending at 24.
        """
        success_I_bins = np.arange(17.0, 24.0 + 0.01, 0.5)
        success_I_centers = (success_I_bins[1:] + success_I_bins[:-1]) / 2.0
        success_rate_dir = self.config.success_rate_dir
        success_I_rate = np.loadtxt(
            os.path.join(success_rate_dir, "VVDSf02_I_success.txt")
        )
        # interpolate the success rate as probability of being selected with
        # the probability at I > 24 being 0
        p_success_I = interp1d(
            success_I_centers,
            success_I_rate,
            kind="quadratic",
            bounds_error=False,
            fill_value=(success_I_rate[0], 0.0),
        )
        # Randomly sample objects according to their success rate
        random_draw = self.rng.random(len(data))
        mask = random_draw < p_success_I(data["mag_i_lsst"])
        # Spec-z success rate as function of redshift read of Figure 13a/b in
        # LeFevre+13 for VVDS deep sample. The listing is split by i_AB into
        # ranges (17.5; 22.5] and (22.5; 24.0].
        # NOTE: at z > 1.75 there are only lower limits (due to a lack of
        # spec-z?), thus the success rate is extrapolated as 1.0 at z > 1.75
        success_z_bright_centers, success_z_bright_rate = np.loadtxt(
            os.path.join(success_rate_dir, "VVDSf02_z_bright_success.txt")
        ).T
        success_z_deep_centers, success_z_deep_rate = np.loadtxt(
            os.path.join(success_rate_dir, "VVDSf02_z_deep_success.txt")
        ).T
        # interpolate the success rates as probability of being selected with
        # the probability in the bright bin at z > 1.75 being 1.0 and the deep
        # bin at z > 4.0 being 0.0
        p_success_z_bright = interp1d(
            success_z_bright_centers,
            success_z_bright_rate,
            kind="quadratic",
            bounds_error=False,
            fill_value=(success_z_bright_rate[0], 1.0),
        )
        p_success_z_deep = interp1d(
            success_z_deep_centers,
            success_z_deep_rate,
            kind="quadratic",
            bounds_error=False,
            fill_value=(success_z_deep_rate[0], 0.0),
        )
        # Randomly sample objects according to their success rate
        random_draw = self.rng.random(len(data))
        iterator = zip(
            [
                data[self.config.colnames["i"]] <= 22.5,
                data[self.config.colnames["i"]] > 22.5,
            ],
            [p_success_z_bright, p_success_z_deep],
        )
        for m, p_success_z in iterator:
            mask[m] &= random_draw[m] < p_success_z(
                data[self.config.colnames["redshift"]][m]
            )
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
    It covers an area of 1.7 deg^2 with ~20000 galaxies.
    For zCOSMOS, the data should at least include i band and redshift.
    """

    name = "specselection_zCOSMOS"

    def photometryCut(self, data):
        """
        Photometry cut for zCOSMOS based on Lilly+09.
        NOTE: This only includes zCOSMOS bright.
        update the internal state
        """
        mask = (data[self.config.colnames["i"]] > 15.0) & (
            data[self.config.colnames["i"]] < 22.5
        )
        # 15.0, 22.5
        self.mask &= mask

    def speczSuccess(self, data):
        """
        Spec-z success rate as function of redshift (x) and I_AB (y) read of
        Figure 3 in Lilly+09 for zCOSMOS bright sample.
        """
        success_rate_dir = self.config.success_rate_dir
        x = np.arange(0, 1.4, 0.00587002, dtype=np.float64)
        y = np.arange(18, 22.4, 0.01464226, dtype=np.float64)

        pixels_y = np.searchsorted(y, data[self.config.colnames["i"]])
        pixels_x = np.searchsorted(x, data[self.config.colnames["redshift"]])

        rates = np.loadtxt(os.path.join(success_rate_dir, "zCOSMOS_success.txt"))
        ratio_list = np.zeros(len(pixels_y))
        for i, py in enumerate(pixels_y):
            if (
                (py >= rates.shape[0])
                or (pixels_x[i] >= rates.shape[1])
                or (py == 0)
                or (pixels_x[i] == 0)
            ):
                ratio_list[i] = 0
            else:
                ratio_list[i] = rates[pixels_y[i] - 1][pixels_x[i] - 1]

        randoms = self.rng.uniform(size=data[self.config.colnames["i"]].size)
        mask = randoms <= ratio_list
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
    or HSC, the data should at least include giz bands and redshift.
    """

    name = "specselection_HSC"

    def photometryCut(self, data):
        """
        HSC galaxies were binned in color magnitude space with i-band mag from -2 to 6 and g-z color from 13 to 26.
        """
        mask = (data[self.config.colnames["i"]] > 13.0) & (
            data[self.config.colnames["i"]] < 26.0
        )
        self.mask &= mask
        gz = data[self.config.colnames["g"]] - data[self.config.colnames["z"]]
        mask = (gz > -2.0) & (gz < 6.0)
        self.mask &= mask

    def speczSuccess(self, data):
        """
        HSC galaxies were binned in color magnitude space with i-band mag from -2 to 6 and g-z color from 13 to 26
        200 bins in each direction. The ratio of of galaxies with spectroscopic redshifts (training galaxies) to
        galaxies with only photometry in HSC wide field (application galaxies) was computed for each pixel. We divide
        the data into the same pixels and randomly select galaxies into the training sample based on the HSC ratios
        """
        success_rate_dir = self.config.success_rate_dir
        x_edge = np.linspace(13, 26, 201, endpoint=True)
        y_edge = np.linspace(-2, 6, 201, endpoint=True)

        rates = np.loadtxt(os.path.join(success_rate_dir, "hsc_success.txt"))

        pixels_y = np.searchsorted(
            y_edge, data[self.config.colnames["g"]] - data[self.config.colnames["z"]]
        )
        pixels_x = np.searchsorted(x_edge, data[self.config.colnames["i"]])

        # Do the color-based, percentile-based redshift cut

        percentile_cut = self.config.percentile_cut

        mask_keep = np.ones_like(data[self.config.colnames["i"]])
        if percentile_cut != 100:  # pragma: no cover
            pixels_y_unique = np.unique(pixels_y)
            pixels_x_unique = np.unique(pixels_x)

            for y in pixels_y_unique:
                for x in pixels_x_unique:
                    ind_inpix = np.where((pixels_y == y) * (pixels_x == x))[0]
                    if ind_inpix.size == 0:
                        continue
                    redshifts = data[self.config.colnames["redshift"]][ind_inpix]
                    percentile = np.percentile(redshifts, percentile_cut)
                    ind_remove = ind_inpix[redshifts > percentile]
                    mask_keep[ind_remove] = 0
            self.mask &= mask_keep

        pixels_y = pixels_y - 1
        pixels_x = pixels_x - 1

        ratio_list = np.zeros(len(pixels_y))
        for i, py in enumerate(pixels_y):
            if (py >= rates.shape[0]) or (pixels_x[i] >= rates.shape[1]):
                ratio_list[i] = 0
            else:
                ratio_list[i] = rates[pixels_y[i]][pixels_x[i]]

        randoms = self.rng.uniform(size=data[self.config.colnames["i"]].size)
        mask = randoms <= ratio_list
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
