"""The LSST Model for photometric errors."""

from numbers import Number
from typing import Iterable, List, Optional, Tuple

import numpy as np
from rail.creation.degrader import Degrader


class LSSTErrorModel(Degrader):
    """The LSST Model for photometric errors.

    Implements the error model from the LSST Overview Paper:
    https://arxiv.org/abs/0805.2366
    Note however that this paper gives the high SNR approximation.
    By default, this model uses the more accurate version of the error model
    where Eq. 5 = (N/S)^2, in flux, and the error is Gaussian in flux space.
    There is a flag allowing you to use the high SNR approximation instead.
    See the __init__ docstring.

    In addition, this paper only gives the error estimation for point sources.
    To account for extended elliptical sources, we follow:
    http://arxiv.org/abs/2007.01846
    We apply two methods to estimate the aperture sizes of the sources according
    to the definitation of measured fluxes:
    1) 'auto' model given by http://arxiv.org/abs/2007.01846;
    2) 'gaap' model given by https://arxiv.org/abs/1902.11265
    In order to use these methods, your dataframe of galaxy data must include columns
    named "major" and "minor", which are the semi-major and minor axes of the galaxies
    in arcseconds. For the cosmoDC2 catalog, they are the half-light radii.

    Create an instance by calling the class, then use the instance as a
    callable on pandas DataFrames.
    Example usage:
    errModel = LSSTErrorModel()
    data_with_errs = errModel(data)
    Error model from the LSST Overview Paper:
    https://arxiv.org/abs/0805.2366

    All parameters are optional. To see the default settings, do
    `LSSTErrorModel().config_options()`

    Note that the dictionary bandNames sets the bands for which this model
    calculates photometric errors. The dictionary keys are the band names
    that the error model uses internally to search for parameters, and the
    corresponding dictionary values are the band names as they appear in
    your data set. By default, the LSST bands are named "u", "g", "r", "i", "z",
    and "y". You can use the bandNames dictionary to alias them differently.

    For example, if in your DataFrame, the bands are named lsst_u, lsst_g, etc.
    you can set bandNames = {"u": "lsst_u", "g": "lsst_g", ...},
    and the error model will work automatically.
    You can also add other bands to bandNames. For example, if you want to
    use the same model to calculate photometric errors for Euclid bands, you
    can include {"euclid_y": "euclid_y", "euclid_j": "euclid_j", ...}.
    In this case, you must include the additional information listed below...

    IMPORTANT: For every band in bandNames, you must provide:
        - nVisYr
        - gamma
        - the single-visit 5-sigma limiting magnitude. You can do this either by
          (1) explicitly providing it in the m5 dictionary, or
          (2) by adding the corresponding parameters to Cm, msky, theta,
          and km, in which case the limiting magnitude will be calculated
          for you, using Eq. 6 from the LSST Overview Paper.

    Note if for any bands, you explicitly pass a limiting magnitude in the
    m5 dictionary, the model will use the explicitly passed value,
    regardless of the values in Cm, msky, theta, and km.

    Parameters
    ----------
    bandNames : dict, optional
        A dictionary of bands for which to calculate errors. The dictionary
        keys are the band names that the Error Model uses internally to
        search for parameters, and the corresponding dictionary values
        are the names of those bands as they appear in your data set.
        Can be used to alias the default names of the LSST bands, or to add
        additional bands. See notes above.
    tvis : float, optional
        Exposure time for a single visit
    nYrObs : float, optional
        Number of years of observations
    nVisYr : dict, optional
        Mean number of visits per year in each band
    gamma : dict, optional
        A band dependent parameter defined in the LSST Overview Paper
    airmass : float, optional
        The fiducial airmass
    extendedSource : float, optional
        Constant to add to magnitudes of extended sources.
        The error model is designed to emulate magnitude errors for point
        sources. This constant provides a zeroth order correction accounting
        for the fact that extended sources have larger uncertainties. Note
        this is only meant to account for small, slightly extended sources.
        For typical LSST galaxies, this may be of order ~0.3.
    sigmaSys : float, optional
        The irreducible error of the system in AB magnitudes.
        Set's the minimum photometric error.
    magLim : float, optional
        The dimmest magnitude allowed. All dimmer magnitudes are set to ndFlag.
    ndFlag : float, optional
        The flag for non-detections. All magnitudes greater than magLim (and
        their corresponding errors) will be set to this value.
    m5 : dict, optional
        A dictionary of single visit 5-sigma limiting magnitudes. For any
        bands for which you pass a value in m5, this will be the 5-sigma
        limiting magnitude used, and any values for that band in Cm, msky,
        theta, and km will be ignored.
    Cm : dict, optional
        A band dependent parameter defined in the LSST Overview Paper
    msky : dict, optional
        Median zenith sky brightness in each band
    theta : dict, optional
        Median zenith seeing FWHM (in arcseconds) for each band
    km : dict, optional
        Atmospheric extinction in each band
    highSNR : bool, default=False
        Sets whether you use the high SNR approximation given in the LSST
        Overview Paper. If False, then Eq. 5 from the LSST Error Model is
        used to calculate (N/S)^2 in flux, and errors are Gaussian in flux
        space. If True, then Eq. 5 is used to calculate the squared error
        in magnitude space, and errors are Gaussian in magnitude space.
    A_min : float, default=2.0
        The minimum GAaP aperture size (in arcmin).
    A_max : float, default=0.7
        The maximum GAaP aperture size (in arcmin).
    errorType: string, default="point"
        Should be "point" for point source errors. For errors for extended sources,
        you can choose either "auto" or "gaap". See the top of the docstring for
        more details.
    """

    name = "LSSTErrorModel"
    config_options = Degrader.config_options.copy()
    config_options.update(
        **{
            "bandNames": {  # provided so you can alias the names of the bands
                "u": "u",
                "g": "g",
                "r": "r",
                "i": "i",
                "z": "z",
                "y": "y",
            },
            "tvis": 30.0,  # exposure time for a single visit in seconds
            "nYrObs": 10.0,  # number of years of observations
            "nVisYr": {  # mean number of visits per year in each filter
                "u": 5.6,
                "g": 8.0,
                "r": 18.4,
                "i": 18.4,
                "z": 16.0,
                "y": 16.0,
            },
            "gamma": {  # band dependent parameter
                "u": 0.038,
                "g": 0.039,
                "r": 0.039,
                "i": 0.039,
                "z": 0.039,
                "y": 0.039,
            },
            "airmass": 1.2,  # fiducial airmass
            "extendedSource": 0.0,  # constant added to m5 for extended sources
            "sigmaSys": 0.005,  # expected irreducible error
            "magLim": 30.0,  # dimmest allowed magnitude; dimmer mags set to ndFlag
            "ndFlag": np.nan,  # flag for non-detections (all mags > magLim)
            "m5": {},  # explicit list of m5 limiting magnitudes
            "Cm": {  # band dependent parameter
                "u": 23.09,
                "g": 24.42,
                "r": 24.44,
                "i": 24.32,
                "z": 24.16,
                "y": 23.73,
            },
            "msky": {  # median zenith sky brightness at Cerro Pachon
                "u": 22.99,
                "g": 22.26,
                "r": 21.20,
                "i": 20.48,
                "z": 19.60,
                "y": 18.61,
            },
            "theta": {  # median zenith seeing FWHM, arcseconds
                "u": 0.81,
                "g": 0.77,
                "r": 0.73,
                "i": 0.71,
                "z": 0.69,
                "y": 0.68,
            },
            "km": {  # atmospheric extinction
                "u": 0.491,
                "g": 0.213,
                "r": 0.126,
                "i": 0.096,
                "z": 0.069,
                "y": 0.170,
            },
            "highSNR": False,
            "A_min": 0.7,  # minimum aperture size in arcseconds
            "A_max": 2.0,  # maximum aperture size in arcseconds
            "errorType": "point",
        }
    )

    def __init__(self, args, comm=None):
        Degrader.__init__(self, args, comm=comm)

        # validate the settings
        self._validate_settings()

        # calculate the single-visit 5-sigma limiting magnitudes using the settings
        self._all_m5 = self._calculate_m5()

        # update the limiting magnitudes with any m5s passed
        if self.config.m5 is not None:
            self._all_m5.update(self.config.m5)

    def _validate_settings(self):
        """Validate all the settings."""

        # check that highSNR is boolean
        if not isinstance(self.config["highSNR"], bool):  # pragma: no cover
            raise TypeError("highSNR must be boolean.")

        if self.config["errorType"] not in ["point", "gaap", "auto"]:
            raise ValueError("errorType should be one of 'point', 'gaap', 'auto'")

        # check all the numbers
        for key in [
            "tvis",
            "nYrObs",
            "airmass",
            "extendedSource",
            "sigmaSys",
            "magLim",
            "ndFlag",
            "A_min",
            "A_max",
        ]:
            # check they are numbers
            # note we also check if they're bools and np.nan's because these
            # are both technically numbers
            is_number = isinstance(self.config[key], Number)
            is_bool = isinstance(self.config[key], bool)
            is_nan = np.isnan(self.config[key])
            # ndFlag can be np.nan
            if key == "ndFlag":
                if not (is_number or is_nan) or is_bool:  # pragma: no cover
                    raise TypeError(f"{key} must be a number or NaN.")
            # the others cannot
            else:
                if not is_number or is_nan or is_bool:  # pragma: no cover
                    raise TypeError(f"{key} must be a number.")
            # if they are numbers, check that they are non-negative
            # except for magLim and ndFlag, which can be
            if key not in ["magLim", "ndFlag"]:
                if self.config[key] < 0:
                    raise ValueError(f"{key} must be non-negative.")

        # make sure bandNames is a dictionary
        if not isinstance(self.config["bandNames"], dict):  # pragma: no cover
            raise TypeError(
                "bandNames must be a dictionary where the keys are the names "
                "of the bands as used internally by the error model, and the "
                "values are the names of the bands as present in your data set."
            )

        # remove unnecessary keys from all setting dictionaries
        for key, val in self.config.items():
            if key in ["aliases"]:
                continue
            if isinstance(val, dict):
                self.config[key] = {
                    band: val[band] for band in self.config["bandNames"] if band in val
                }

        # check all the other dictionaries
        for key in ["nVisYr", "gamma", "Cm", "msky", "theta", "km"]:

            # make sure they are dictionaries
            if not isinstance(self.config[key], dict):  # pragma: no cover
                raise TypeError(f"{key} must be a dictionary.")

            # check the values in the dictionary
            for subkey, val in self.config[key].items():

                # check that it's a number
                is_number = isinstance(val, Number)
                is_bool = isinstance(val, bool)
                if not is_number or is_bool:
                    raise TypeError(f"{key}['{subkey}'] must be a number.")

                # for certain dictionaries, check that the numbers are positive
                if key not in ["Cm", "msky"]:
                    if val < 0:
                        raise ValueError(f"{key}['{subkey}'] must be positive.")

            # get the set of bands in bandNames that aren't in this dictionary
            missing = set(self.config["bandNames"]) - set(self.config[key])

            # nVisYr and gamma must have an entry for every band in bandNames
            if key in ["nVisYr", "gamma"]:
                if len(missing) > 0:
                    raise ValueError(
                        f"{key} must have an entry for every band in bandNames, "
                        f"and is currently missing entries for {missing}."
                    )

            # but for the other dictionaries...
            else:
                # we don't need an entry for every band in bandNames, as long
                # as the missing bands are listed in m5
                missing -= set(self.config["m5"])

                if len(missing) > 0:
                    raise ValueError(
                        "You haven't provided enough information to calculate "
                        f"limiting magnitudes for {missing}. Please include "
                        f"entries for these bands in {key}, or explicitly pass "
                        "their 5-sigma limiting magnitudes in m5."
                    )

    def _calculate_m5(self) -> dict:
        """Calculate the single-visit m5 limiting magnitudes.

        Uses Eq. 6 from https://arxiv.org/abs/0805.2366
        Note this is only done for the bands for which an m5 wasn't explicitly passed.
        """

        # get the settings
        settings = self.config

        # get the list of bands for which an m5 wasn't explicitly passed
        bands = set(self.config["bandNames"]) - set(self.config["m5"])
        bands = [band for band in self.config["bandNames"] if band in bands]

        # calculate the m5 limiting magnitudes using Eq. 6
        m5 = {
            band: settings["Cm"][band]
            + 0.50 * (self.config["msky"][band] - 21)
            + 2.5 * np.log10(0.7 / settings["theta"][band])
            + 1.25 * np.log10(settings["tvis"] / 30)
            - settings["km"][band] * (settings["airmass"] - 1)
            - settings["extendedSource"]
            for band in bands
        }

        return m5

    def _get_bands_and_names(
        self, columns: Iterable[str]
    ) -> Tuple[List[str], List[str]]:
        """Get the bands and bandNames that are present in the given data columns."""

        # get the list of bands present in the data
        bandNames = list(set(self.config["bandNames"].values()).intersection(columns))

        # sort bandNames to be in the same order provided in settings["bandNames"]
        bandNames = [
            band for band in self.config["bandNames"].values() if band in bandNames
        ]

        # get the internal names of the bands from bandNames
        bands = [
            {bandName: band for band, bandName in self.config["bandNames"].items()}[
                bandName
            ]
            for bandName in bandNames
        ]

        return bands, bandNames

    def _get_area_ratio_gaap(
        self, majors: np.ndarray, minors: np.ndarray, bands: list
    ) -> np.ndarray:
        """Get the ratio between PSF area and galaxy aperture area for "gaap" model."""
        # get the psf size for each band
        theta_size = np.array([self.config["theta"][band] for band in bands])

        hl_to_sigma = 1 / 0.6745  # half IQR to Gaussian sigma
        fwhm_to_sigma = 1 / 2.355

        # convert PSF FWHM to a Gaussian sigma
        theta_sigma = theta_size * fwhm_to_sigma
        A_min_sigma = self.config["A_min"] * fwhm_to_sigma
        A_max_sigma = self.config["A_max"] * fwhm_to_sigma

        # calculate the area of the psf in each band
        A_psf = np.pi * theta_sigma**2

        # convert the half-light radii to the sigma of semi-major and minor axis
        majors *= hl_to_sigma
        minors *= hl_to_sigma

        # calculate the area of the galaxy aperture in each band
        a_ap = np.sqrt(
            theta_sigma[None, :] ** 2 + majors[:, None] ** 2 + A_min_sigma**2
        )
        a_ap[a_ap > A_max_sigma] = A_max_sigma
        b_ap = np.sqrt(
            theta_sigma[None, :] ** 2 + minors[:, None] ** 2 + A_min_sigma**2
        )
        b_ap[b_ap > A_max_sigma] = A_max_sigma
        A_ap = np.pi * a_ap * b_ap

        # return their ratio
        return A_ap / A_psf

    def _get_area_ratio_auto(
        self, majors: np.ndarray, minors: np.ndarray, bands: list
    ) -> np.ndarray:
        """Get the ratio between PSF area and galaxy aperture area for "auto" model."""
        # get the psf size for each band
        theta_size = np.array([self.config["theta"][band] for band in bands])
        fwhm_to_sigma = 1 / 2.355

        # convert PSF FWHM to a Gaussian sigma
        theta_sigma = theta_size * fwhm_to_sigma
        # calculate the area of the psf in each band
        A_psf = np.pi * theta_sigma**2

        # calculate the area of the galaxy aperture in each band
        a_ap = np.sqrt(theta_size[None, :] ** 2 + (2.5 * majors[:, None]) ** 2)
        b_ap = np.sqrt(theta_size[None, :] ** 2 + (2.5 * minors[:, None]) ** 2)
        A_ap = np.pi * a_ap * b_ap

        # return their ratio
        return A_ap / A_psf

    def _get_NSR(self, mags: np.ndarray, bands: list) -> np.ndarray:
        """Calculate the noise-to-signal ratio.

        Uses Eqs 4 and 5 from https://arxiv.org/abs/0805.2366
        """

        # get the 5-sigma limiting magnitudes for these bands
        m5 = np.array([self._all_m5[band] for band in bands])
        # and the values for gamma
        gamma = np.array([self.config["gamma"][band] for band in bands])

        # calculate x as defined in the paper
        x = 10 ** (0.4 * np.subtract(mags, m5))

        # calculate the squared NSR for a single visit
        # Eq. 5 in https://arxiv.org/abs/0805.2366
        nsrRandSqSingleExp = (0.04 - gamma) * x + gamma * x**2

        # calculate the random NSR for the stacked image
        nVisYr = np.array([self.config["nVisYr"][band] for band in bands])
        nStackedObs = nVisYr * self.config["nYrObs"]
        nsrRand = np.sqrt(nsrRandSqSingleExp / nStackedObs)

        # get the irreducible system NSR
        if self.config["highSNR"]:
            nsrSys = self.config["sigmaSys"]
        else:
            nsrSys = 10 ** (self.config["sigmaSys"] / 2.5) - 1

        # calculate the total NSR
        nsr = np.sqrt(nsrRand**2 + nsrSys**2)

        return nsr

    def _get_obs_and_errs(
        self,
        mags: np.ndarray,
        bands: list,
        seed: Optional[int],
        majors: np.ndarray = None,
        minors: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return observed magnitudes and magnitude errors."""

        rng = np.random.default_rng(seed)

        # get the aperture-psf ratio for all the galaxies
        if self.config["errorType"] == "gaap":
            a_ratio = self._get_area_ratio_gaap(majors, minors, bands)
        elif self.config["errorType"] == "auto":
            a_ratio = self._get_area_ratio_auto(majors, minors, bands)
        else:
            a_ratio = 1

        # get the NSR for all the galaxies
        nsr = self._get_NSR(mags, bands) * np.sqrt(a_ratio)

        if self.config["highSNR"]:
            # in the high SNR approximation, mag err ~ nsr, and we can
            # model errors as Gaussian in magnitude space

            # calculate observed magnitudes
            obsMags = rng.normal(loc=mags, scale=nsr)

            # decorrelate the magnitude errors from the true magnitudes
            obsMagErrs = self._get_NSR(obsMags, bands) * np.sqrt(a_ratio)

        else:
            # in the more accurate error model, we acknowledge err != nsr,
            # and we model errors as Gaussian in flux space

            # calculate observed magnitudes
            fluxes = 10 ** (mags / -2.5)
            obsFluxes = fluxes * (1 + rng.normal(scale=nsr))
            obsFluxes = np.clip(obsFluxes, 0, None)
            with np.errstate(divide="ignore"):
                obsMags = -2.5 * np.log10(obsFluxes)

            # decorrelate the magnitude errors from the true magnitudes
            obsFluxNSR = self._get_NSR(obsMags, bands) * np.sqrt(a_ratio)
            obsMagErrs = 2.5 * np.log10(1 + obsFluxNSR)

        # flag magnitudes beyond magLim as non-detections
        idx = np.where(obsMags > self.config["magLim"])
        obsMags[idx] = self.config["ndFlag"]
        obsMagErrs[idx] = self.config["ndFlag"]

        return obsMags, obsMagErrs

    def run(self):
        """Calculate errors for data, and save the results in a pandas DataFrame."""
        # get the bands and bandNames present in the data
        data = self.get_data("input", allow_missing=True)
        bands, bandNames = self._get_bands_and_names(data.columns)

        # get numpy array of magnitudes
        mags = data[bandNames].to_numpy()

        if self.config["errorType"] != "point":
            # get the sizes of the galaxies
            majors = data["major"].to_numpy()
            minors = data["minor"].to_numpy()

            # get observed magnitudes and magnitude errors
            obsMags, obsMagErrs = self._get_obs_and_errs(
                mags, bands, self.config.seed, majors, minors
            )
        else:
            # get observed magnitudes and magnitude errors
            obsMags, obsMagErrs = self._get_obs_and_errs(mags, bands, self.config.seed)

        # save the observations in a DataFrame
        obsData = data.copy()
        obsData[bandNames] = obsMags
        obsData[[band + "_err" for band in bandNames]] = obsMagErrs

        # re-order columns so that the error columns come right after the
        # respective magnitudes
        columns = data.columns.tolist()
        for band in bandNames:
            columns.insert(columns.index(band) + 1, band + "_err")
        obsData = obsData[columns]
        self.add_data("output", obsData)

    def get_limiting_mags(
        self,
        Nsigma: float = 5,
        coadded: bool = False,
    ) -> dict:
        """Return the limiting magnitudes for all bands.

        This method essentially reverse engineers the _get_NSR() method
        so that we calculate what magnitude results in NSR = 1/Nsigma.
        (NSR is noise-to-signal ratio; NSR = 1/SNR)

        Parameters
        ----------
        Nsigma : float, default=5
            Sets which limiting magnitude to return, e.g. Nsigma = 1 returns
            the 1-sigma limiting magnitude. In other words, Nsigma is equal
            to the signal-to-noise ratio (SNR) of the limiting magnitudes.
        coadded : bool, default=False
            Whether to return the limiting magnitudes for a single visit,
            or for a coadded image.
        """

        # get the bands and bandNames
        bands, bandNames = self._get_bands_and_names(self.config["bandNames"].values())
        # get the 5-sigma limiting magnitudes for these bands
        m5 = np.array([self._all_m5[band] for band in bands])

        # and the values for gamma
        gamma = np.array([self.config["gamma"][band] for band in bands])

        # get the number of exposures
        if coadded:
            nVisYr = np.array([self.config["nVisYr"][band] for band in bands])
            nStackedObs = nVisYr * self.config["nYrObs"]
        else:
            nStackedObs = 1

        # get the irreducible system error
        if self.config["highSNR"]:
            nsrSys = self.config["sigmaSys"]
        else:
            nsrSys = 10 ** (self.config["sigmaSys"] / 2.5) - 1

        # calculate the square of the random NSR that a single exposure must have
        nsrRandSqSingleExp = (1 / Nsigma**2 - nsrSys**2) * nStackedObs

        # calculate the value of x that corresponds to this NSR
        # note this is just the quadratic equation,
        # applied to NSR^2 = (0.04 - gamma) * x + gamma * x^2
        x = (
            (gamma - 0.04)
            + np.sqrt((gamma - 0.04) ** 2 + 4 * gamma * nsrRandSqSingleExp)
        ) / (2 * gamma)

        # convert x to a limiting magnitude
        limiting_mags = m5 + 2.5 * np.log10(x)

        # return as a dictionary
        return dict(zip(bandNames, limiting_mags))

    def __repr__(self):  # pragma: no cover
        """Define how the model is represented and printed."""

        settings = self.config

        # start message
        printMsg = "LSSTErrorModel parameters:\n\n"

        # list all bands
        printMsg += "Model for bands: "
        printMsg += ", ".join(settings["bandNames"].values()) + "\n"

        # print whether using the high SNR approximation
        if self.config["highSNR"]:
            printMsg += "Using the high SNR approximation\n\n"
        else:
            printMsg += "\n"

        # print which error mode we are using
        printMsg += f"Using error type {settings['errorType']}\n"

        # exposure time
        printMsg += f"Exposure time = {settings['tvis']} s\n"
        # number of years
        printMsg += f"Number of years of observations = {settings['nYrObs']}\n"
        # mean visits per year
        printMsg += "Mean visits per year per band:\n   "
        printMsg += (
            ", ".join(
                [
                    f"{bandName}: {settings['nVisYr'][band]}"
                    for band, bandName in settings["bandNames"].items()
                ]
            )
            + "\n"
        )
        # airmass
        printMsg += f"Airmass = {settings['airmass']}\n"
        # irreducible error
        printMsg += f"Irreducible system error = {settings['sigmaSys']}\n"
        # extended sources
        if settings["extendedSource"] > 0:
            printMsg += f"Extended source model: add {settings['extendedSource']}"
            printMsg += "mag to 5-sigma depth for point sources\n"
        # minimum magnitude
        printMsg += f"Magnitudes dimmer than {settings['magLim']} are "
        printMsg += f"set to {settings['ndFlag']}\n"
        # gamma
        printMsg += "gamma for each band:\n   "
        printMsg += (
            ", ".join(
                [
                    f"{bandName}: {settings['gamma'][band]}"
                    for band, bandName in settings["bandNames"].items()
                ]
            )
            + "\n\n"
        )

        # coadded m5's
        printMsg += "The coadded 5-sigma limiting magnitudes are:\n"
        printMsg += (
            ", ".join(
                [
                    f"{bandName}: {limiting_mag:.2f}"
                    for bandName, limiting_mag in self.get_limiting_mags(
                        coadded=True
                    ).items()
                ]
            )
            + "\n\n"
        )

        # explicit m5
        if len(settings["m5"]) > 0:
            printMsg += "The following single-visit 5-sigma limiting magnitudes "
            printMsg += "were explicitly passed:\n   "
            printMsg += (
                ", ".join(
                    [
                        f"{bandName}: {settings['m5'][band]}"
                        for band, bandName in settings["bandNames"].items()
                        if band in settings["m5"]
                    ]
                )
                + "\n\n"
            )

        m5 = self._calculate_m5()
        if len(m5) > 0:
            # Calculated m5
            printMsg += "The following single-visit 5-sigma limiting magnitudes are\n"
            printMsg += "calculated using the parameters that follow them:\n   "
            printMsg += (
                ", ".join(
                    [
                        f"{settings['bandNames'][band]}: {val:.2f}"
                        for band, val in m5.items()
                    ]
                )
                + "\n"
            )
            # Cm
            printMsg += "Cm for each band:\n   "
            printMsg += (
                ", ".join(
                    [
                        f"{settings['bandNames'][band]}: {val}"
                        for band, val in settings["Cm"].items()
                    ]
                )
                + "\n"
            )
            # msky
            printMsg += "Median zenith sky brightness in each band:\n   "
            printMsg += (
                ", ".join(
                    [
                        f"{settings['bandNames'][band]}: {val}"
                        for band, val in settings["msky"].items()
                    ]
                )
                + "\n"
            )
            # theta
            printMsg += "Median zenith seeing FWHM (in arcseconds) for each band:\n   "
            printMsg += (
                ", ".join(
                    [
                        f"{settings['bandNames'][band]}: {val}"
                        for band, val in settings["theta"].items()
                    ]
                )
                + "\n"
            )
            # km
            printMsg += "Extinction coefficient for each band:\n   "
            printMsg += (
                ", ".join(
                    [
                        f"{settings['bandNames'][band]}: {val}"
                        for band, val in settings["km"].items()
                    ]
                )
                + "\n"
            )

        return printMsg
