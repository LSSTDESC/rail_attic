""" Abstract base class defining a degrader

The key feature is that the __call__ method takes a pandas DataFrame and a
seed, and returns a pandas DataFrame, and wraps the run method
"""

import pandas as pd

from rail.core.stage import RailStage
from rail.core.data import PqHandle

class Degrader(RailStage):
    """Base class Degraders, which apply various degradations to synthetic photometric data"""

    name = 'Degrader'
    config_options = RailStage.config_options.copy()
    config_options.update(seed=12345)
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, comm=None):
        """Initialize Degrader that can degrade photometric data"""
        RailStage.__init__(self, args, comm=comm)

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
        self.finalize()
        return self.get_handle('output')
