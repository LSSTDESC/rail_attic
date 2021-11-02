# Abstract base class defining a degrader

# note that this is just to indicate that any callable object can serve as
# a degrader, as long as its __call__ method takes a pandas DataFrame and a
# seed, and returns a pandas DataFrame

from abc import ABC, abstractmethod

import pandas as pd


class Degrader(ABC):
    @abstractmethod
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
