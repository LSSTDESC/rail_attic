# Abstract base class defining a degrader

# note that this is just to indicate that any callable object can serve as
# a degrader, as long as its __call__ method takes a pandas DataFrame and a
# seed, and returns a pandas DataFrame

import pandas as pd

from rail.core.stage import RailStage
from rail.core.data import TableHandle

class Degrader(RailStage):

    name = 'Degrader'
    config_options = dict(seed=12345)
    inputs = [('input', TableHandle)]    
    outputs = [('output', TableHandle)]

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)
        self._cached = None

    @property
    def cached(self):
        return self._cached
        
    def __call__(self, sample: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """Return a degraded sample.

        Parameters
        ----------
        sample : pd.DataFrame
            The sample to be degraded
        seed : int, default=None
            An integer to set the numpy random seed

        Returns
        -------
        pd.DataFrame
            The degraded sample
        """
        if seed is not None:
            self.config.seed = seed
        self.set_data('input', sample)
        self.run()
        return self._cached
