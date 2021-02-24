# Abstract base class defining an engine,
# which represents a probability distribution

# note this is just to indicate that any object with these methods can be
# used as the engine of the Creator object. You can subclass Engine to
# wrap whatever object you want for Creator.
# see flowEngine.py for an example of a wrapper for pzflow.

from abc import ABC, abstractmethod


class Engine(ABC):
    def __init__(self):
        """Initialize Engine that can sample galaxy data and draw posteriors."""

    @abstractmethod
    def sample(self, n_samples, seed=None, **kwargs):
        """Return a random sample of the distribution with size n_samples."""

    @abstractmethod
    def get_posterior(self, data, column, grid, **kwargs):
        """Return posteriors for the given column over the values in grid."""
