import numpy as np
import pandas as pd
from rail.creation.engines import Engine
from typing import Callable


class Creator:
    """Object that supplies mock data for redshift estimation experiments.

    The mock data is drawn from a probability distribution defined by the
    generator, with an optional degrader applied.
    """

    def __init__(self, engine: Engine, degrader: Callable = None, info: dict = None):
        """
        Parameters
        ----------
        engine: rail.Engine object
            Object defining a redshift probability distribution.
            Must have sample, log_prob and get_posterior methods (see engine.py)
        degrader: callable, optional
            A function or other callable that degrades the generated sample.
            Must take a pandas DataFrame and a seed number, and return a
            pandas DataFrame representing the degraded sample.
        info: any, optional
            Additional information desired to be stored with the instance
            as a dictionary.
        """
        self.engine = engine
        self.degrader = degrader
        self.info = info

    def get_posterior(self, data: pd.DataFrame, column: str, grid: np.ndarray):
        """Calculate the posterior of the given column over the values in grid.

        Parameters
        ----------
        data : pd.DataFrame
            Pandas dataframe of the data on which the posteriors are conditioned.
        column : str
            Name of the column for which the posterior is calculated.
        grid : np.ndarray
            Grid over which the posterior is calculated.

        Returns
        -------
        np.ndarray
            Array of posteriors, of shape (data.shape[0], grid.size).
        """
        return self.engine.get_posterior(data, column, grid)

    def sample(
        self,
        n_samples: int,
        seed: int = None,
        include_pdf: bool = False,
        pz_grid: np.ndarray = None,
    ):
        """Draws n_samples from the engine

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        seed : int, optional
            sets the random seed for drawing samples
        include_pdf : boolean, optional
            If True, redshift posteriors are returned for each galaxy.
            The posteriors are saved in the column pz_pdf, and the
            redshift grid saved as df.attrs['pz_grid'].
        pz_grid : np.array, default=np.arange(0, 2.02, 0.02)
            The grid over which to calculate the redshift posteriors.

        Returns
        -------
        outputs : pd.DataFrame
            samples from model, containing photometry, true redshift, and
            redshift posterior PDF's if requested.

        Notes
        -----
        Output posterior format is currently hardcoded to grid evaluations but could be integrated with qp.
        We will probably change the output format to dovetail with the evaluation module when ready.
        """

        if include_pdf is True and pz_grid is None:
            pz_grid = np.arange(0, 2.02, 0.02)

        rng = np.random.default_rng(seed)

        # get samples
        outputs = self.engine.sample(n_samples, seed=seed)

        if self.degrader is not None:
            # degrade sample
            outputs = self.degrader(outputs, seed=seed)
            # calculate fraction that survives the cut
            selected_frac = len(outputs) / n_samples
            # draw more samples and degrade until we have enough samples
            while len(outputs) < n_samples:
                # estimate how many extras to draw
                n_supplement = int(1.1 / selected_frac * (n_samples - len(outputs)))
                # draw new samples and apply cut
                new_sample = self.engine.sample(n_supplement, seed=rng.integers(1e18))
                new_sample = self.degrader(new_sample, seed=rng.integers(1e18))
                # add these to the larger set
                outputs = pd.concat((outputs, new_sample), ignore_index=True)
            # cut out the extras
            outputs = outputs[:n_samples]

        # calculate posteriors
        if include_pdf:
            posteriors = self.get_posterior(outputs, column="redshift", grid=pz_grid)
            outputs.attrs["pz_grid"] = pz_grid
            outputs["pz_pdf"] = list(posteriors)

        return outputs
