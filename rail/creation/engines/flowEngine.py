# this is a subclass of Engine that wraps a pzflow Flow so that it can
# be used as the engine of a Creator object.

import pandas as pd
import qp
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

    def sample(self, n_samples: int, seed: int = None) -> pd.DataFrame:
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

    def get_posterior(
        self,
        data: pd.DataFrame,
        column: str,
        grid: np.ndarray,
        marg_rules: dict = None,
        err_samples: int = None,
        seed: int = None,
        batch_size: int = None,
        nan_to_zero: bool = True,
    ) -> qp.Ensemble:
        """Calculate posteriors with the pzflow Flow.

        Parameters
        ----------
        data : pd.DataFrame
            Pandas dataframe of the data on which the posteriors are conditioned.
            Must have all columns in self.flow.data_columns, *except*
            for the column specified for the posterior (see below).
        column : str
            Name of the column for which the posterior is calculated.
            Must be one of the columns in self.flow.data_columns. However,
            whether or not this column is present in `data` is irrelevant.
        grid : np.ndarray
            Grid over which the posterior is calculated.
        marg_rules : dict, optional
            Dictionary with rules for marginalizing over missing variables.
            The dictionary must contain the key "flag", which gives the flag
            that indicates a missing value. E.g. if missing values are given
            the value 99, the dictionary should contain {"flag": 99}.
            The dictionary must also contain {"name": callable} for any
            variables that will need to be marginalized over, where name is
            the name of the variable, and callable is a callable that takes
            the row of variables and returns a grid over which to marginalize
            the variable. E.g. {"y": lambda row: np.linspace(0, row["x"], 10)}.
            Note: the callable for a given name must *always* return an array
            of the same length, regardless of the input row.
        err_samples : int, optional
            Number of samples from the error distribution to average over for
            the posterior calculation. If provided, Gaussian errors are assumed,
            and method will look for error columns in `inputs`. Error columns
            must end in `_err`. E.g. the error column for the variable `u` must
            be `u_err`. Zero error assumed for any missing error columns.
        seed: int, optional
            Random seed for drawing samples from the error distribution.
        batch_size: int, default=None
            Size of batches in which to calculate posteriors. If None, all
            posteriors are calculated simultaneously. This is faster, but
            requires more memory.
        nan_to_zero : bool, default=True
            Whether to convert NaN's to zero probability in the final pdfs.

        Returns
        -------
        qp.Ensemble
            A qp.Ensemble of pdfs, linearly interpolated on the grid
        """
        pdfs = self.flow.posterior(
            inputs=data,
            column=column,
            grid=grid,
            marg_rules=marg_rules,
            err_samples=err_samples,
            seed=seed,
            batch_size=batch_size,
            nan_to_zero=nan_to_zero,
        )
        return qp.Ensemble(qp.interp, data={"xvals": grid, "yvals": pdfs})
