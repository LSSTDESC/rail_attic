"""
Abstract base classes defining a Creator, which will create synthetic photometric data
and a PosteriorCalculator, which can calculate posteriors for the data with respect
to the distribution defined by the creator.
"""

import pandas as pd
import qp
from rail.core.stage import RailStage

class Modeler(RailStage):
    """
    Base class for creating a model of redshift and photometry.

    """
    name = 'Modeler'
    config_options = RailStage.config_options.copy()
    config_options.update(seed=12345)

    def __init__(self, args, comm=None):
        """Initialize Modeler"""
        RailStage.__init__(self, args, comm=comm)

    def fit_model(self, **kwargs):
        """
        Produce a creation model from which photometry and redshifts can be generated

        Parameters
        ----------
        [The parameters depend entirely on the modeling approach!]

        Returns
        -------
        [This will definitely be a file, but the filetype and format depend entirely on the modeling approach!]
        """
        config_options = RailStage.config_options.copy()
        inputs = [('base', DataHandle)]
        outputs = [('model', ModelHandle)]
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        return self.get_handle('model')

class Creator(RailStage):
    """Base class for Creators that generate synthetic photometric data from a model.

    `Creator` will output a table of photometric data.  The details 
    will depend on the particular engine.
    """

    name = 'Creator'
    config_options = RailStage.config_options.copy()
    config_options.update(n_samples=int, seed=12345)

    def __init__(self, args, comm=None):
        """Initialize Creator"""
        RailStage.__init__(self, args, comm=comm)

    def sample(self, n_samples: int, seed: int = None, **kwargs) -> pd.DataFrame:
        """Draw samples from the model specified in the configuration.

        This is a method for running a Creator in interactive mode.
        In pipeline mode, the subclass `run` method will be called by itself.

        Parameters
        ----------
        n_samples: int
            The number of samples to draw
        seed: int
            The random seed to control sampling

        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame of the samples

        Notes
        -----
        This method puts `n_samples` and `seed` into the stage configuration
        data, which makes them available to other methods.
        It then calls the `run` method, which must be defined by a subclass.
        Finally, the `DataHandle` associated to the `output` tag is returned. 
        """
        self.config['n_samples'] = n_samples
        self.config['seed'] = seed
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        return self.get_handle('output')


class PosteriorCalculator(RailStage):
    """Base class for object that calculates the posterior distribution of a
    particular field in a table of photometric data  (typically the redshift).

    The posteriors will be contained in a qp Ensemble.
    """

    name = 'PosteriorCalculator'
    config_options = RailStage.config_options.copy()
    config_options.update(column=str)

    def __init__(self, args, comm=None):
        """Initialize PosteriorCalculator """
        RailStage.__init__(self, args, comm=comm)

    def get_posterior(self, data: pd.DataFrame, column: str, **kwargs) -> qp.Ensemble:
        """Return posteriors for the given column.

        This is a method for running a Creator in interactive mode.
        In pipeline mode, the subclass `run` method will be called by itself.

        Parameters
        ----------
        data: pd.DataFrame
            A Pandas DataFrame of the galaxies for which posteriors are calculated
        column: str
            The name of the column in the DataFrame for which posteriors are calculated

        Notes
        -----
        This will put the `data` argument input this Stages the DataStore using this stages `input` tag.
        This will put the additional functional arguments into this Stages configuration data.

        It will then call `self.run()` and return the `DataHandle` associated to the `output` tag
        """
        self.set_data('input', data)
        self.config.update(column=column)
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        return self.get_handle('output')
