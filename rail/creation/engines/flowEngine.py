# this is a subclass of Engine that wraps a pzflow Flow so that it can
# be used as the engine of a Creator object.

import numpy as np
import pandas as pd
import qp

from collections import UserDict

from pzflow import Flow

from rail.core.data import TableHandle, QPHandle
from rail.core.types import DataFile
from rail.creation.engines import Engine, PosteriorEvaluator


class FlowDict(dict):
    """ 
    A specialized dict to keep track of individual flow objects: this is just a dict these additional features

    1. Keys are paths 
    2. Values are flow objects, this is checked at runtime.
    3. There is a read(path, force=False) method that reads a flow object and inserts it into the dictionary
    4. There is a single static instance of this class    
    """

    def __setitem__(self, key, value):
        if not isinstance(value, Flow):
            raise TypeError(f"Only values of type Flow can be added to a FlowFactory, not {type(value)}")
        return dict.__setitem__(self, key, value)

    def read(self, path, force=False):
        if force or path not in self:
            flow = Flow(file=path)
            self.__setitem__(path, flow)
            return flow
        return self[path]


FLOW_FACTORY = FlowDict()

def FlowFactory():
    return FLOW_FACTORY


class FlowFile(DataFile):
    """
    A file that describes a PZFlow object
    """

    @classmethod
    def open(cls, path, mode, **kwargs):
        return FLOW_FACTORY.read(path)


class FlowEngine(Engine):
    """Engine wrapper for a pzflow Flow object."""

    name = 'FlowEngine'
    inputs = [('flow_file', FlowFile)]
    outputs = [('output', TableHandle)]
    
    def __init__(self, args, comm=None):
        Engine.__init__(self, args, comm=comm)
        self.flow = None
        self.open_flow(**args)

    def open_flow(self, **kwargs):
        """Instantiate a pzflow Flow engine.

        Keywords
        --------
        flow : pzflow Flow object
            A trained pzflow Flow from which the creator will sample
            and will use to draw posteriors.
        flow_file : str
            A file from which to load a Flow object
        """
        flow = kwargs.get('flow', None)
        if flow is not None:
            self.flow = flow
            self.config['flow_file'] = None
            return
        flow_file = kwargs.get('flow_file', None)
        if flow_file is not None:
            self.config['flow_file'] = flow_file
            self.flow = self.open_input('flow_file')

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
        self.config.n_samples = n_samples
        if seed is not None:
            self.config.seed = seed
        self.run()
        return self._cached
        
    def run(self):
        if self.flow is None:
            raise ValueError("Tried to run a FlowEngine before the Flow object is loaded")
        self._cached = self.add_data('output', self.flow.sample(self.config.n_samples, self.config.seed))
        

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
    config_options.update(grid=np.ndarray,
                          err_samples=10,
                          seed=12345,
                          marg_rules={"flag": np.nan, "mag_u_lsst": lambda row: np.linspace(25, 31, 10)},
                          batch_size=10000,
                          nan_to_zero=True)        
        
    inputs = [('flow_file', FlowFile),
              ('input', TableHandle)]
    outputs = [('output', QPHandle)]

    def __init__(self, args, comm=None):
        PosteriorEvaluator.__init__(self, args, comm=comm)
        self.flow = None
        self.open_flow(**args)
    
    def open_flow(self, **kwargs):
        """Instantiate a pzflow Flow engine.

        Keywords
        --------
        flow : pzflow Flow object
            A trained pzflow Flow from which the creator will sample
            and will use to draw posteriors.
        flow_file : str
            A file from which to load a Flow object
        """
        flow = kwargs.get('flow', None)
        if flow is not None:
            self.flow = flow
            self.config['flow_file'] = None
            return
        flow_file = kwargs.get('flow_file', None)
        if flow_file is not None:
            self.config['flow_file'] = flow_file
            self.flow = self.open_input('flow_file')
    
    def run(self):
        data = self.get_data('input')
        if self.config.marg_rules is None:
            marg_rules = {"flag": np.nan, "mag_u_lsst": lambda row: np.linspace(25, 31, 10)}
        else:
            marg_rules = self.config.marg_rules

        pdfs = self.flow.posterior(
            inputs=data,
            column=self.config.column,
            grid=self.config.grid,
            err_samples=self.config.err_samples,
            seed=self.config.seed,
            marg_rules=self.config.marg_rules,
            batch_size=self.config.batch_size,
            nan_to_zero=self.config.nan_to_zero,
        )

        ensemble = qp.Ensemble(qp.interp, data={"xvals": self.config.grid, "yvals": pdfs})
        self._cached = self.add_data('output', ensemble)
        
