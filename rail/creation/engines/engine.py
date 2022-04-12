"""
Abstract base classes defining an Engine, which will create synthetic photometric data
and a PosteriorEvaluator, which can construction Posterior distributions associated to
that data
"""

import pandas as pd
import qp

from rail.core.stage import RailStage

class Engine(RailStage):
    """Base class for Engines that create synthetic photometric data

    `Engine` will output a table of photometric data.  The details
    will depend on the particular engine.
    """

    name = 'Engine'
    config_options = RailStage.config_options.copy()
    config_options.update(n_samples=int, seed=12345)

    def __init__(self, args, comm=None):
        """Initialize Engine"""
        RailStage.__init__(self, args, comm=comm)

    def sample(self, n_samples: int, seed: int = None, **kwargs) -> pd.DataFrame:
        """The main interface method for the photo-z creation `Engine`

        Return a random sample of the distribution with size n_samples.

        Notes
        -----
        This will put the keyword command line arguments into the stage configuraiton data.
        That will make it available to the other methods.
        It will then call `self.run()` and return the `DataHandle` associated to the `output` tag
        """
        self.config['n_samples'] = n_samples
        self.config['seed'] = seed
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        return self.get_handle('output')


class PosteriorEvaluator(RailStage):
    """Base class for object that evaluate the posterior distribution of a
    particular field in a table of photometric data  (typically the redshift)
    """

    name = 'PosteriorEvaluator'
    config_options = RailStage.config_options.copy()
    config_options.update(column=str)

    def __init__(self, args, comm=None):
        """Initialize PosteriorEvaluator """
        RailStage.__init__(self, args, comm=comm)

    def get_posterior(self, data: pd.DataFrame, column: str, **kwargs) -> qp.Ensemble:
        """Return posteriors for the given column over the values in grid.

        Notes
        -----
        This will put the `data` argument input this Stages the DataStore using this stages `input` tag.
        This will put the additonal functional arguments into this Stages configuration data.

        It will then call `self.run()` and return the `DataHandle` associated to the `output` tag
        """
        self.set_data('input', data)
        self.config.update(column=column)
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        return self.get_handle('output')
