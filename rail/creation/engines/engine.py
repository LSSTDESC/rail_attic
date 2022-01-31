# Abstract base class defining an engine, which represents a probability distribution

# note this is just to indicate that any object with these methods can be
# used as the engine of the Creator object. You can subclass Engine to
# wrap whatever object you want for Creator.

from abc import ABC, abstractmethod

import pandas as pd
import qp

from rail.core.stage import RailStage

class Engine(RailStage):

    name = 'Engine'
    config_options = dict(n_samples=int, seed=12345)
    
    def __init__(self, args, comm=None):
        """Initialize Engine that can sample galaxy data and draw posteriors."""
        RailStage.__init__(self, args, comm=comm)
        self._cached = None

    @property
    def cached(self):
        return self._cached
        
    def sample(self, n_samples: int, seed: int = None, **kwargs) -> pd.DataFrame:
        """Return a random sample of the distribution with size n_samples."""
        self.config['n_samples'] = n_samples
        self.config['seed'] = seed
        self.config.update(**kwargs)
        self.run()


class PosteriorEvaluator(RailStage):
    
    name = 'PosteriorEvaluator'
    config_options = dict(column=str)

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)
        self._cached = None

    @property
    def cached(self):
        return self._cached

    def get_posterior(self, data: pd.DataFrame, column: str, **kwargs) -> qp.Ensemble:
        """Return posteriors for the given column over the values in grid."""
        self.set_data(self.get_aliased_tag('input'), data)
        self.config.update(column=column)
        self.config.update(**kwargs)
        self.run()
        return self._cached
