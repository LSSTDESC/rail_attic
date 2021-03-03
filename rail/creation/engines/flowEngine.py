# this is a subclass of Engine that wraps a pzflow Flow so that it can
# be used as the engine of a Creator object.

from rail.creation.engines import Engine


class FlowEngine(Engine):
    """Engine wrapper for a pzflow Flow object."""

    def __init__(self, flow):
        """Instantiate a pzflow Flow engine.

        Parameters
        ----------
        flow : pzflow Flow object
            The trained pzflow Flow from which the creator will sample
            and will use to draw posteriors.
        """
        self.flow = flow

    def sample(self, n_samples: int, seed: int = None):
        """Sample from the pzflow Flow.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        seed : int, optional
            sets the random seed for drawing samples
        Returns
        -------
        outputs : pd.DataFrame
            samples from the Flow.
        """
        return self.flow.sample(nsamples=n_samples, seed=seed)

    def get_posterior(self, data, column, grid, **kwargs):
        """Calculate posteriors with the pzflow Flow.

        Parameters
        ----------
        data : pd.DataFrame
            Pandas dataframe of the data on which the posteriors are conditioned.
        column : str
            Name of the column for which the posterior is calculated.
        grid : np.ndarray
            Grid over which the posterior is calculated.

        Returns
        -------
        np.ndarray
            Array of posteriors, of shape (data.shape[0], grid.size).
        """
        return self.flow.posterior(data, column=column, grid=grid)
