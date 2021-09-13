import numpy as np
import pandas as pd


class LineConfusion:
    """Degrader that simulates emission line confusion.

    Example: degrader = LineConfusion(true_wavelen=3727,
                                      wrong_wavelen=5007,
                                      frac_wrong=0.05)
    is a degrader that misidentifies 5% of OII lines (at 3727 angstroms)
    as OIII lines (at 5007 angstroms), which results in a larger
    spectroscopic redshift .

    Note that when selecting the galaxies for which the lines are confused,
    the degrader ignores galaxies for which this line confusion would result
    in a negative redshift, which can occur for low redshift galaxies when
    wrong_wavelen < true_wavelen.
    """

    def __init__(self, true_wavelen: float, wrong_wavelen: float, frac_wrong: float):
        """
        Parameters
        ----------
        true_wavelen : positive float
            The wavelength of the true emission line.
            Wavelength unit assumed to be the same as wrong_wavelen.
        wrong_wavelen : positive float
            The wavelength of the wrong emission line, which is being confused
            for the correct emission line.
            Wavelength unit assumed to be the same as true_wavelen.
        frac_wrong : float between zero and one
            The fraction of galaxies with confused emission lines.
        """

        # convert to floats
        true_wavelen = float(true_wavelen)
        wrong_wavelen = float(wrong_wavelen)
        frac_wrong = float(frac_wrong)

        # validate parameters
        if true_wavelen < 0:
            raise ValueError("true_wavelen must be positive")
        if wrong_wavelen < 0:
            raise ValueError("wrong_wavelen must be positive")
        if frac_wrong < 0 or frac_wrong > 1:
            raise ValueError("frac_wrong must be between 0 and 1.")

        self.true_wavelen = true_wavelen
        self.wrong_wavelen = wrong_wavelen
        self.frac_wrong = frac_wrong

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame of galaxy data to be degraded.
        seed : int, default=None
            Random seed for the degrader.

        Returns
        -------
        pd.DataFrame
            DataFrame of the degraded galaxy data.
        """

        # convert to an array for easy manipulation
        values, columns = data.values.copy(), data.columns.copy()
        # get the minimum redshift
        # if wrong_wavelen < true_wavelen, this is minimum the redshift for
        # which the confused redshift is still positive
        zmin = self.wrong_wavelen / self.true_wavelen - 1
        # select the random fraction of galaxies whose lines are confused
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            np.where(values[:, 0] > zmin)[0],
            size=int(self.frac_wrong * values.shape[0]),
            replace=False,
        )
        # transform these redshifts
        values[idx, 0] = (
            1 + values[idx, 0]
        ) * self.true_wavelen / self.wrong_wavelen - 1
        # return results in a data frame
        return pd.DataFrame(values, columns=columns)


class InvRedshiftIncompleteness:
    """Degrader that simulates incompleteness with a selection function
    inversely proportional to redshift.

    The survival probability of this selection function is
    p(z) = min(1, z_p/z),
    where z_p is the pivot redshift.
    """

    def __init__(self, pivot_redshift):
        """
        Parameters
        ----------
        pivot_redshift : positive float
            The redshift at which the incompleteness begins.
        """
        pivot_redshift = float(pivot_redshift)
        if pivot_redshift < 0:
            raise ValueError("pivot redshift must be positive.")

        self.pivot_redshift = pivot_redshift

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame of galaxy data to be degraded.
        seed : int, default=None
            Random seed for the degrader.

        Returns
        -------
        pd.DataFrame
            DataFrame of the degraded galaxy data.
        """

        # calculate survival probability for each galaxy
        survival_prob = np.clip(self.pivot_redshift / data["redshift"], 0, 1)
        # probabalistically drop galaxies from the data set
        rng = np.random.default_rng(seed)
        idx = np.where(rng.random(size=data.shape[0]) <= survival_prob)
        return data.iloc[idx]
