from typing import Callable, Union

import numpy as np
import pandas as pd
import qp
from rail.core.multi import MultiStage
from rail.creation.degradation import Degrader
#from rail.creation.engines import Engine


class Creator(MultiStage):
    """Object that supplies mock data for redshift estimation experiments.

    The mock data is drawn from a probability distribution defined by the
    generator, with an optional degrader applied.
    """

    def __init__(self, engine, degraders, **kwargs):
        """
        Parameters
        ----------
        engine: rail.Engine object
            Object defining a redshift probability distribution.
            Must have sample, log_prob and get_posterior methods (see engine.py)
        degraders: list of rail.Degrader
            A Degrader, function, or other callable that degrades the generated
            sample. Must take a pandas DataFrame and a seed int, and return a
            pandas DataFrame representing the degraded sample.
        """
        MultiStage.__init__(self, **kwargs)
        self.engine = engine
        self.degraders = degraders        
        self.add_stage(engine)
        for degrader_ in self.degraders:
            self.add_stage(degrader_)
            
        
    def get_posterior(
        self, data: pd.DataFrame, column: str = "redshift", **kwargs
    ) -> qp.Ensemble:
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

    def set_config(self, n_samples, seed=None):
        self.engine.config.n_samples = n_samples
        if seed is None:
            return
        for degrader_ in self.degraders:
            degrader.config.seed = seed            
    
    def sample(self, n_samples: int, seed: int = None) -> pd.DataFrame:
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
        self.set_config(n_samples, seed)
        self.run()
        if self.degraders:
            last_stage = self.degraders[-1]
        else:
            last_stage = self.engine
        return last_stage.cached
        
