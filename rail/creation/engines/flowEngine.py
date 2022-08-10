""" This is the subclass of Engine that wraps a pzflow Flow so that it can
used to generate synthetic data """

import numpy as np
import qp

from rail.core.data import PqHandle, QPHandle, FlowHandle
from rail.creation.engines.engine import Engine, PosteriorEvaluator



class FlowEngine(Engine):
    """Engine wrapper for a pzflow Flow object."""

    name = 'FlowEngine'
    inputs = [('flow', FlowHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, comm=None):
        """ Constructor

        Does standard Engine initialization and also gets the `Flow` object
        """
        Engine.__init__(self, args, comm=comm)
        if not isinstance(args, dict):
            args = vars(args)
        self.set_flow(**args)

    def set_flow(self, **kwargs):
        """ Set the flow, either from an object or by loading from a file """
        flow = kwargs.get('flow')
        if flow is None:  #pragma: no cover
            return None
        from pzflow import Flow
        if isinstance(flow, Flow):
            return self.set_data('flow', flow)
        return self.set_data('flow', data=None, path=flow)

    def run(self):
        """ Run method

        Calls `Flow.sample` to use the `Flow` object to generate photometric data

        Notes
        -----
        Puts the data into the data store under this stages 'output' tag
        """
        flow = self.get_data('flow')
        if flow is None:  #pragma: no cover
            raise ValueError("Tried to run a FlowEngine before the Flow object is loaded")
        self.add_data('output', flow.sample(self.config.n_samples, seed=self.config.seed))


class FlowPosterior(PosteriorEvaluator):
    """Engine wrapper for a pzflow Flow object

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
    err_samples : int, optional
        Number of samples from the error distribution to average over for
        the posterior calculation. If provided, Gaussian errors are assumed,
        and method will look for error columns in `inputs`. Error columns
        must end in `_err`. E.g. the error column for the variable `u` must
        be `u_err`. Zero error assumed for any missing error columns.
    seed: int, optional
        Random seed for drawing samples from the error distribution.
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
        DEFAULT: the default marg_rules dict is
        {
        "flag": np.nan,
        "u": np.linspace(25, 31, 10),
        }
    batch_size: int, default=None
        Size of batches in which to calculate posteriors. If None, all
        posteriors are calculated simultaneously. This is faster, but
        requires more memory.
    nan_to_zero : bool, default=True
        Whether to convert NaN's to zero probability in the final pdfs.

    """

    name = 'FlowPosterior'
    config_options = PosteriorEvaluator.config_options.copy()
    config_options.update(grid=list,
                          err_samples=10,
                          seed=12345,
                          marg_rules={"flag": np.nan, "mag_u_lsst": lambda row: np.linspace(25, 31, 10)},
                          batch_size=10000,
                          nan_to_zero=True)

    inputs = [('flow', FlowHandle),
              ('input', PqHandle)]
    outputs = [('output', QPHandle)]

    def __init__(self, args, comm=None):
        """ Constructor

        Does standard Engine initialization and also gets the `Flow` object
        """
        PosteriorEvaluator.__init__(self, args, comm=comm)

    def run(self):
        """ Run method

        Calls `Flow.posterior` to use the `Flow` object to get the posterior disrtibution

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Puts the data into the data store under this stages 'output' tag
        """

        data = self.get_data('input')
        flow = self.get_data('flow')
        if self.config.marg_rules is None:  #pragma: no cover
            marg_rules = {"flag": np.nan, "mag_u_lsst": lambda row: np.linspace(25, 31, 10)}
        else:
            marg_rules = self.config.marg_rules

        pdfs = flow.posterior(
            inputs=data,
            column=self.config.column,
            grid=np.array(self.config.grid),
            err_samples=self.config.err_samples,
            seed=self.config.seed,
            marg_rules=marg_rules,
            batch_size=self.config.batch_size,
            nan_to_zero=self.config.nan_to_zero,
        )

        ensemble = qp.Ensemble(qp.interp, data={"xvals": self.config.grid, "yvals": pdfs})
        self.add_data('output', ensemble)
