from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from rail.creation.degradation import Degrader


class LSSTErrorModel(Degrader):
    """
    LSST Model for photometric errors.

    Implements the error model from https://arxiv.org/abs/0805.2366

    Instantiated as a class object, then used as a callable.
    Takes a pandas DataFrame as input.

    Example usage:
    errModel = LSSTErrorModel()
    data_with_errs = errModel(data)
    """

    def __init__(
        self,
        bandNames: dict = None,
        tvis: float = None,
        nYrObs: float = None,
        nVisYr: dict = None,
        gamma: dict = None,
        airmass: float = None,
        extendedSource: float = None,
        sigmaSys: float = None,
        minMag: float = None,
        m5: dict = None,
        Cm: dict = None,
        msky: dict = None,
        theta: dict = None,
        km: dict = None,
    ):
        """
        All parameters are optional. To see the default settings, do
        `LSSTErrorModel().default_settings()`

        Note that the dictionary bandNames sets the bands for which this model
        calculates photometric errors. The dictionary keys are the band names
        that the error model uses internally to search for parameters, and the
        corresponding dictionary values are the band names as they appear in
        your data set. By default, the LSST bands are named "lsst_u", "lsst_g",
        "lsst_r", "lsst_i", "lsst_z", and "lsst_y". You can use the bandNames
        dictionary to alias them differently.

        For example, if in your DataFrame, the bands are named u, g, r, i ,z, y,
        you can set bandNames = {"lsst_u": "u", "lsst_g": "g", ...},
        and the error model will work automatically.

        You can also add other bands to bandNames. For example, if you want to
        use the same model to calculate photometric errors for Euclid bands, you
        can include {"euclid_y": "euclid_y", "euclid_j": "euclid_j", ...}.
        In this case, you must include the additional information listed below...

        IMPORTANT: For every band in bandNames, you must provide:
            - nVisYr
            - gamma
            - the 5-sigma limiting magnitude. You can do this either by
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
            A dictionary of bands for which to calculate errors. The dictioary
            keys are the band names that the Error Model uses internally to
            search for parameters, and the corresponding dictionary values
            are the names of those bands as they appear in your data set.

            Can be used to
            alias the default names of the LSST bands, or to add additional bands.
            See notes above.
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
            Constant to add to magnitudes of extended sources
        sigmaSys : float, optional
            The irreducible error of the system. Set's the minimum photometric error.
        minMag : float, optional
            The dimmest magnitude allowed. All dimmer magnitudes are set to 99.
        m5 : dict, optional
            A dictionary of 5-sigma limiting magnitudes. For any bands for which
            you pass a value in m5, this will be the 5-sigma limiting magnitude
            used, and any values for that band in Cm, msky, theta, and km will
            be ignored.
        Cm : dict, optional
            A band dependent parameter defined in the LSST Overview Paper
        msky : dict, optional
            Median zenith sky brightness in each band
        theta : dict, optional
            Median zenith seeing FWHM (in arcseconds) for each band
        km : dict, optional
            Atmospheric extinction in each band
        """

        # update the settings
        self.settings = self.default_settings()
        if bandNames is not None:
            self.settings["bandNames"] = bandNames
        if tvis is not None:
            self.settings["tvis"] = tvis
        if nYrObs is not None:
            self.settings["nYrObs"] = nYrObs
        if nVisYr is not None:
            self.settings["nVisYr"] = nVisYr
        if gamma is not None:
            self.settings["gamma"] = gamma
        if airmass is not None:
            self.settings["airmass"] = airmass
        if extendedSource is not None:
            self.settings["extendedSource"] = extendedSource
        if sigmaSys is not None:
            self.settings["sigmaSys"] = sigmaSys
        if minMag is not None:
            self.settings["minMag"] = minMag
        if Cm is not None:
            self.settings["Cm"] = Cm
        if msky is not None:
            self.settings["msky"] = msky
        if theta is not None:
            self.settings["theta"] = theta
        if km is not None:
            self.settings["km"] = km
        if m5 is not None:
            # make sure it's a dictionary
            assert isinstance(m5, dict), "m5 must be a dictionary, or None."
            # save m5
            self.settings["m5"] = m5
            # remove these bands from the dictionaries that hold information
            # about how to calculate m5
            for key1 in ["Cm", "msky", "theta", "km"]:
                for key2 in m5:
                    self.settings[key1].pop(key2, None)

        # validate the settings
        self.validate_settings()

        # calculate the 5-sigma limiting magnitudes using the settings
        self.m5 = self.calculate_m5()

        # update the limiting magnitudes with any m5s passed
        if m5 is not None:
            self.m5.update(m5)

    def default_settings(self):
        """
        Default settings for the error model. All of these values come from
        the LSST Overview Paper, https://arxiv.org/abs/0805.2366

        Each setting is defined with an inline comment, and the location of
        the number in the paper is designated with the following codes:
            pN - page number N
            T1 - Table 1, on page 11
            T2 - Table 2, on page 26
        """
        return {
            "bandNames": {  # provided so you can alias the names of the bands
                "lsst_u": "lsst_u",
                "lsst_g": "lsst_g",
                "lsst_r": "lsst_r",
                "lsst_i": "lsst_i",
                "lsst_z": "lsst_z",
                "lsst_y": "lsst_y",
            },
            "tvis": 30.0,  # exposure time for a single visit in seconds, p12
            "nYrObs": 10.0,  # number of years of observations
            "nVisYr": {  # mean number of visits per year in each filter (T1)
                "lsst_u": 5.6,
                "lsst_g": 8.0,
                "lsst_r": 18.4,
                "lsst_i": 18.4,
                "lsst_z": 16.0,
                "lsst_y": 16.0,
            },
            "gamma": {  # band dependent parameter (T2)
                "lsst_u": 0.038,
                "lsst_g": 0.039,
                "lsst_r": 0.039,
                "lsst_i": 0.039,
                "lsst_z": 0.039,
                "lsst_y": 0.039,
            },
            "airmass": 1.2,  # fiducial airmass (T2)
            "extendedSource": 0.0,  # constant added to m5 for extended sources
            "sigmaSys": 0.005,  # expected irreducible error, p26
            "minMag": 30.0,  # dimmest allowed magnitude; dimmer mags set to 99
            "m5": {},  # explicit list of m5 limiting magnitudes
            "Cm": {  # band dependent parameter (T2)
                "lsst_u": 23.09,
                "lsst_g": 24.42,
                "lsst_r": 24.44,
                "lsst_i": 24.32,
                "lsst_z": 24.16,
                "lsst_y": 23.73,
            },
            "msky": {  # median zenith sky brightness at Cerro Pachon (T2)
                "lsst_u": 22.99,
                "lsst_g": 22.26,
                "lsst_r": 21.20,
                "lsst_i": 20.48,
                "lsst_z": 19.60,
                "lsst_y": 18.61,
            },
            "theta": {  # median zenith seeing FWHM, arcseconds (T2)
                "lsst_u": 0.81,
                "lsst_g": 0.77,
                "lsst_r": 0.73,
                "lsst_i": 0.71,
                "lsst_z": 0.69,
                "lsst_y": 0.68,
            },
            "km": {  # atmospheric extinction (T2)
                "lsst_u": 0.491,
                "lsst_g": 0.213,
                "lsst_r": 0.126,
                "lsst_i": 0.096,
                "lsst_z": 0.069,
                "lsst_y": 0.170,
            },
        }

    def validate_settings(self):
        """
        Validate all the settings.
        """

        # check all the floats
        for key in [
            "tvis",
            "nYrObs",
            "airmass",
            "extendedSource",
            "sigmaSys",
            "minMag",
        ]:
            # check they are floats
            assert isinstance(self.settings[key], float), f"{key} must be a float."
            # check they are non-negative
            if key != "minMag":
                assert self.settings[key] >= 0, f"{key} must be non-negative."

        # check all the dictionaries
        for key in ["bandNames", "nVisYr", "gamma", "Cm", "msky", "theta", "km"]:

            # make sure they are dictionaries
            assert isinstance(self.settings[key], dict), f"{key} must be a dictionary."

            # get the set of bands in bandNames that aren't in this dictionary
            missing = set(self.settings["bandNames"]) - set(self.settings[key])

            # nVisYr and gamma must have an entry for every band in bandNames
            if key == "nVisYr" or key == "gamma":
                assert len(missing) == 0, (
                    f"{key} must have an entry for every band in bandNames, "
                    f"and is currently missing entries for {missing}."
                )
            # but for the other dictionaries...
            else:

                # we dont need an entry for every band in bandNames, as long as
                # the missing bands are listed in m5
                missing -= set(self.settings["m5"])

                assert len(missing) == 0, (
                    "You haven't provided enough information to calculate limiting "
                    f"magnitudes for {missing}. Please include entries for these bands "
                    f"in {key}, or explicity pass their 5-sigma limiting magnitudes "
                    "in m5."
                )

    def calculate_m5(self) -> dict:
        """
        Calculate the m5 limiting magnitudes,
        using Eq. 6 from https://arxiv.org/abs/0805.2366

        Note this is only done for the bands for which an m5 wasn't
        explicitly passed.
        """

        # get the settings
        settings = self.settings

        # get the list of bands for which an m5 wasn't explicitly passed
        bands = set(self.settings["bandNames"]) - set(self.settings["m5"])

        # calculate the m5 limiting magnitudes using Eq. 6
        m5 = {
            band: settings["Cm"][band]
            + 0.50 * (self.settings["msky"][band] - 21)
            + 2.5 * np.log10(0.7 / settings["theta"][band])
            + 1.25 * np.log10(settings["tvis"] / 30)
            - settings["km"][band] * (settings["airmass"] - 1)
            - settings["extendedSource"]
            for band in bands
        }

        return m5

    def getBandsAndNames(self, columns: Iterable[str]) -> Tuple[List[str], List[str]]:
        """
        Get the bands and bandNames that are present in the given data columns.
        """

        # get the list of bands present in the data
        bandNames = list(set(self.settings["bandNames"].values()).intersection(columns))

        # sort bandNames to be in the same order provided in settings["bandNames"]
        bandNames = [
            band for band in self.settings["bandNames"].values() if band in bandNames
        ]

        # get the internal names of the bands from bandNames
        bands = [
            {bandName: band for band, bandName in self.settings["bandNames"].items()}[
                bandName
            ]
            for bandName in bandNames
        ]

        return bands, bandNames

    def getMagError(self, mags: np.ndarray, bands: list) -> np.ndarray:
        """
        Calculate the magnitude errors using Eqs 4 and 5 from
        https://arxiv.org/abs/0805.2366
        """

        # get the 5-sigma limiting magnitudes for these bands
        m5 = np.array([self.m5[band] for band in bands])
        # and the values for gamma
        gamma = np.array([self.settings["gamma"][band] for band in bands])

        # calculate x as defined in the paper
        x = 10 ** (0.4 * (mags - m5))

        # calculate the squared random error for a single visit
        # Eq. 5 in https://arxiv.org/abs/0805.2366
        sigmaRandSqSingleExp = (0.04 - gamma) * x + gamma * x ** 2

        # calculate the random error for the stacked image
        nVisYr = np.array([self.settings["nVisYr"][band] for band in bands])
        nStackedObs = nVisYr * self.settings["nYrObs"]
        sigmaRand = np.sqrt(sigmaRandSqSingleExp / nStackedObs)

        # calculate total photometric errors
        # Eq. 4 in https://arxiv.org/abs/0805.2366
        sigma = np.sqrt(self.settings["sigmaSys"] ** 2 + sigmaRand ** 2)

        return sigma

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """
        Calculate errors for data, and save the results in a pandas DataFrame.
        """

        # get the bands and bandNames present in the data
        bands, bandNames = self.getBandsAndNames(data.columns)

        # get numpy array of magnitudes
        mags = data[bandNames].to_numpy()

        # calculate the magnitude error
        magErrs = self.getMagError(mags, bands)

        # convert mags to fluxes
        fluxes = 10 ** (mags / -2.5)
        fluxErrs = np.log(10) / 2.5 * fluxes * magErrs

        # add Gaussian flux error
        rng = np.random.default_rng(seed)
        obsFluxes = rng.normal(loc=fluxes, scale=fluxErrs)

        # only fluxes above minFlux will have observed magnitudes recorded
        # everything else is marked as a non-detection
        minFlux = 10 ** (self.settings["minMag"] / -2.5)
        idx = np.where(obsFluxes > minFlux)

        # convert fluxes back to magnitudes
        obsMags = np.full(obsFluxes.shape, 99.0)
        obsMags[idx] = -2.5 * np.log10(obsFluxes[idx])

        # decorrelate the magnitude error
        obsMagErrs = np.full(obsMags.shape, 99.0)
        obsMagErrs[idx] = self.getMagError(obsMags, bands)[idx]

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

        return obsData

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """

        settings = self.settings

        # start message
        printMsg = "LSSTErrorModel parameters:\n\n"

        # list all bands
        printMsg += f"Model for bands: "
        for band in settings["bandNames"].values():
            printMsg += band + ", "
        printMsg = printMsg[:-2] + "\n\n"

        # exposure time
        printMsg += f"Exposure time = {settings['tvis']} s\n"
        # number of years
        printMsg += f"Number of years of observations = {settings['nYrObs']}\n"
        # mean visits per year
        printMsg += "Mean visits per year per band:\n   "
        for i in [
            f"{bandName}: {settings['nVisYr'][band]}, "
            for band, bandName in settings["bandNames"].items()
        ]:
            printMsg += i
        printMsg = printMsg[:-2] + "\n"
        # airmass
        printMsg += f"Airmass = {settings['airmass']}\n"
        # irreducible error
        printMsg += f"Irreducible system error = {settings['sigmaSys']}\n"
        # extended sources
        printMsg += f"Extended source model: add {settings['extendedSource']}"
        printMsg += "mag to 5-sigma depth for point sources\n"
        # minimum magnitude
        printMsg += f"Magnitudes dimmer than {settings['minMag']} are set to 99\n"
        # gamma
        printMsg += "gamma for each band:\n   "
        for i in [
            f"{bandName}: {settings['gamma'][band]}, "
            for band, bandName in settings["bandNames"].items()
        ]:
            printMsg += i
        printMsg = printMsg[:-2] + "\n\n"

        # explicit m5
        if len(settings["m5"]) > 0:
            printMsg += (
                "The following 5-sigma limiting mags were explicitly passed:\n   "
            )
            for i in [
                f"{bandName}: {settings['m5'][band]}, "
                for band, bandName in settings["bandNames"].items()
            ]:
                printMsg += i
            printMsg = printMsg[:-2] + "\n\n"

        m5 = self.calculate_m5()
        if len(m5) > 0:
            # Calculated m5
            printMsg += "The following 5-sigma limiting mags are calculated using "
            printMsg += "the parameters that follow them:\n   "
            for i in [
                f"{settings['bandNames'][band]}: {val:.2f}, "
                for band, val in m5.items()
            ]:
                printMsg += i
            printMsg = printMsg[:-2] + "\n"
            # Cm
            printMsg += "Cm for each band:\n   "
            for i in [
                f"{settings['bandNames'][band]}: {val}, "
                for band, val in settings["Cm"].items()
            ]:
                printMsg += i
            printMsg = printMsg[:-2] + "\n"
            # msky
            printMsg += "Median zenith sky brightness in each band:\n   "
            for i in [
                f"{settings['bandNames'][band]}: {val}, "
                for band, val in settings["msky"].items()
            ]:
                printMsg += i
            printMsg = printMsg[:-2] + "\n"
            # theta
            printMsg += "Median zenith seeing FWHM (in arcseconds) for each band:\n   "
            for i in [
                f"{settings['bandNames'][band]}: {val}, "
                for band, val in settings["theta"].items()
            ]:
                printMsg += i
            printMsg = printMsg[:-2] + "\n"
            # km
            printMsg += "Extinction coefficient for each band:\n   "
            for i in [
                f"{settings['bandNames'][band]}: {val}, "
                for band, val in settings["km"].items()
            ]:
                printMsg += i
            printMsg = printMsg[:-2] + "\n"

        return printMsg
