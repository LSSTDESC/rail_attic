""" Abstract base class defining a degrader

The key feature is that the __call__ method takes a pandas DataFrame and a
seed, and returns a pandas DataFrame, and wraps the run method
"""

import pandas as pd

from rail.core.stage import RailStage
from rail.core.data import PqHandle

class Degrader(RailStage):
    """Base class Degraders, which apply various degradations to synthetic photometric data

    Degraders take "input" data in the form of pandas dataframes in Parquet files and
    provide as "output" another pandas dataframes written to Parquet files
    """

    name = 'Degrader'
    config_options = RailStage.config_options.copy()
    config_options.update(seed=12345)
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, comm=None):
        """Initialize Degrader that can degrade photometric data"""
        RailStage.__init__(self, args, comm=comm)

    def __call__(self, sample: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """The main interface method for `Degrader`

        Applies degradation.

        This will attach the sample to this `Degrader`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Estimator
        by using `self.add_data('output', output_data)`.

        Finally, this will return a PqHandle providing access to that output data.

        Parameters
        ----------
        sample : pd.DataFrame
            The sample to be degraded
        seed : int, default=None
            An integer to set the numpy random seed

        Returns
        -------
        output_data : `PqHandle`
            A handle giving access to a pd.DataFrame with degraded sample
        """
        if seed is not None:
            self.config.seed = seed
        self.set_data('input', sample)
        self.run()
        self.finalize()
        return self.get_handle('output')
