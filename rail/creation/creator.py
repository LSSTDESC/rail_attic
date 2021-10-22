import numpy as np
import pandas as pd
import qp
from rail.creation.degradation import Degrader
from rail.creation.engines import Engine


class Creator:
    """Object that supplies mock data for redshift estimation experiments.

    The mock data is drawn from a probability distribution defined by the
    generator, with an optional degrader applied.
    """

    def __init__(self, engine: Engine, degrader: Degrader = None, info: dict = None):
        """
        Parameters
        ----------
        engine: rail.Engine object
            Object defining a redshift probability distribution.
            Must have sample, log_prob and get_posterior methods (see engine.py)
        degrader: callable, optional
            A Degrader, function, or other callable that degrades the generated
            sample. Must take a pandas DataFrame and a seed int, and return a
            pandas DataFrame representing the degraded sample.
        info: any, optional
            Additional information desired to be stored with the instance
            as a dictionary.
        """
        self.engine = engine
        self.degrader = degrader
        self.info = info

    def get_posterior(self, data: pd.DataFrame, column: str, **kwargs) -> qp.Ensemble:
        """Calculate the posterior of the given column over the values in grid.

        Parameters
        ----------
        data : pd.DataFrame
            Pandas dataframe of the data on which the posteriors are conditioned.
        column : str
            Name of the column for which the posterior is calculated.
        kwargs
            Keyword arguments to pass to the engine get_posterior method

        Returns
        -------
        qp.Ensemble
            A qp.Ensemble of pdfs
        """
        return self.engine.get_posterior(data, column, **kwargs)

    def sample(
        self,
        n_samples: int,
        seed: int = None,
    ) -> pd.DataFrame:
        """Draws n_samples from the engine

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        seed : int, optional
            sets the random seed for drawing samples

        Returns
        -------
        outputs : pd.DataFrame
            samples from the model
        """
        rng = np.random.default_rng(seed)

        # get samples
        outputs = self.engine.sample(n_samples, seed=rng.integers(1e18))

        if self.degrader is not None:
            # degrade the sample
            outputs = self.degrader(outputs, seed=rng.integers(1e18))
            # calculate the fraction that survives the cut
            selected_frac = len(outputs) / n_samples
            # draw more samples and degrade until we have enough samples
            while len(outputs) < n_samples:
                # estimate how many extra galaxies to draw
                n_supplement = int(1.1 / selected_frac * (n_samples - len(outputs)))
                # draw new samples and apply cut
                new_sample = self.engine.sample(n_supplement, seed=rng.integers(1e18))
                new_sample = self.degrader(new_sample, seed=rng.integers(1e18))
                # add these to the larger set
                outputs = pd.concat((outputs, new_sample), ignore_index=True)
            # cut out the extras
            outputs = outputs.iloc[:n_samples, :]

        return outputs
