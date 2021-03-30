# Abstract base class defining an engine, which represents a probability distribution

# note this is just to indicate that any object with these methods can be
# used as the engine of the Creator object. You can subclass Engine to
# wrap whatever object you want for Creator.

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Engine(ABC):
    def __init__(self):
        """Initialize Engine that can sample galaxy data and draw posteriors."""

    @abstractmethod
    def sample(self, n_samples: int, seed: int = None, **kwargs) -> pd.DataFrame:
        """Return a random sample of the distribution with size n_samples."""

    @abstractmethod
    def get_posterior(self, data: pd.DataFrame, column: str, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Return posteriors for the given column over the values in grid."""
