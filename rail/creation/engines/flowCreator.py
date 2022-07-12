"""This is the subclass of Creator that wraps a PZFlow Flow so that it can
be used to generate synthetic data and calculate posteriors."""

from ast import Param
from sre_constants import MAGIC
from xml.etree.ElementInclude import LimitedRecursiveIncludeError

import numpy as np
import qp
from pzflow import Flow

from rail.core.data import FlowHandle, PqHandle, QPHandle, TableHandle
from rail.creation.creator import Creator, Modeler, PosteriorCalculator

def newrange(mins, maxs):
    newmins, newmaxs = [], []
    for i in range(len(mins) - 1):
        newmins.append(mins[i] - maxs[i+1])
        newmaxs.append(maxs[i] + mins[i+1])
    return(newmins, newmaxs)

class FlowModeler(Modeler):
    """Modeler wrapper for a PZFlow Flow object.

    This class trains the flow.
    """

    name = "FlowModeler"
    inputs = [("catalog", TableHandle)]  # move this to the base class!!!!!!
    outputs = [("flow", FlowHandle)]

    config_options = Modeler.config_options.copy()
    config_options.update(
        column_names=Param(
            dict,
            {'phys_cols': ['redshift'], 'phot_cols': ['u', 'g', 'r', 'i', 'z', 'y']},
            msg="The column names of the input data.",
        ),
        calc_colors=Param(
            bool,
            False,
            msg="Do you provide magnitudes but want to model colors?",
        ),
        # ref_column_name=Param(
        #     str,
        #     'r',
        #     msg="The column name for the magnitude to use as an anchor for defining colors.",
        # ),
        data_mins=Param(
            list,
            None, # should we default this?
            msg=(
                "The minima of the values modeled by the flow. The list must "
                "be in the same order of the columns you are modeling."
            ),
        ),
        data_maxs=Param(
            list,
            None,  # should we default this?
            msg=(
                "The maxima of the values modeled by the flow. The list must "
                "be in the same order of the columns you are modeling."
            ),
        ),
        spline_knots=Param(
            int,
            16,
            msg="The number of spline knots in the normalizing flow.",
        ),
    )

    def __init__(self, args, comm=None):
        """Constructor

        Does standard Modeler initialization.
        """

        # first let's pull out the ranges of each column of the data
        mins = self.config.data_mins
        maxs = self.config.data_maxs

        # now let's set up the RQ-RSC
        nlayers = len(self.config.column_names) # can make configurable from yaml if wanted
        transformed_dim = 1 # can make configurable from yaml if wanted
        K = self.config.spline_knots

        # if we are doing the color transform, there are a few more things to do...
        if color_config := self.config["calc_colors"]:
            # tell it which column to use as the reference magnitude
            ref_idx = train_set.columns.get_loc(color_config["ref_col"])
            # and which columns correspond to the magnitudes we want colors for
            mag_idx = [columns.get_loc(band) for band in color_config["bands"]]

            # convert ranges above to the corresponding color ranges
            # CONVERT MAG MINS AND MAXES TO COLOR MINS AND MAXES
            color_cols =

            # chain all the bijectors together
            bijector = Chain(
                ColorTransform(ref_idx, mag_idx),
                ShiftBounds(mins, maxs),
                RollingSplineCoupling(nlayers, K=K, transformed_dim=transformed_dim),
            )
        # otherwise, we can just chain what we have
        else:
            bijector = Chain(
                ShiftBounds(mins, maxs),
                RollingSplineCoupling(nlayers, K=K, transformed_dim=transformed_dim),
            )

        # build the flow
        flow = Flow(train_set.columns, bijector=bijector)

    def run(self):
        """

        """
        if self.config.base:
            training_data = self.get_data('base')[self.config.base]
        else:  #pragma:  no cover
            training_data = self.get_data('base')

        if self.config.num_training_epochs  :
            n_epoch = self.get_data('input')[self.config.num_training_epochs]
        else:  #pragma:  no cover
            n_epoch = self.get_data('input')


        # train the flow
        losses = flow.train(training_data, epochs=n_epoch, verbose=True)

        # save the flow
        self.add_data("output", flow)
        flow.save(self.config.model_file)


class FlowCreator(Creator):
    """Creator wrapper for a PZFlow Flow object."""

    name = "FlowCreator"
    inputs = [("flow", FlowHandle)]
    outputs = [("output", PqHandle)]

    def __init__(self, args, comm=None):
        """Constructor

        Does standard Creator initialization and also gets the `Flow` object
        """
        Creator.__init__(self, args, comm=comm)
        if not isinstance(args, dict):
            args = vars(args)
        self.set_flow(**args)

    def set_flow(self, **kwargs):
        """Set the `Flow`, either from an object or by loading from a file."""
        flow = kwargs.get("flow")
        if flow is None:  # pragma: no cover
            return None
        from pzflow import Flow

        if isinstance(flow, Flow):
            return self.set_data("flow", flow)
        return self.set_data("flow", data=None, path=flow)

    def run(self):
        """Run method

        Calls `Flow.sample` to use the `Flow` object to generate photometric data

        Notes
        -----
        Puts the data into the data store under this stages 'output' tag
        """
        flow = self.get_data("flow")
        if flow is None:  # pragma: no cover
            raise ValueError(
                "Tried to run a FlowCreator before the `Flow` model is loaded"
            )
        self.add_data("output", flow.sample(self.config.n_samples, self.config.seed))


class FlowPosterior(PosteriorCalculator):
    """PosteriorCalculator wrapper for a PZFlow Flow object

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

    name = "FlowPosterior"
    config_options = PosteriorCalculator.config_options.copy()
    config_options.update(
        grid=list,
        err_samples=10,
        seed=12345,
        marg_rules={"flag": np.nan, "mag_u_lsst": lambda row: np.linspace(25, 31, 10)},
        batch_size=10000,
        nan_to_zero=True,
    )

    inputs = [("flow", FlowHandle), ("input", PqHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args, comm=None):
        """Constructor

        Does standard PosteriorCalculator initialization
        """
        PosteriorCalculator.__init__(self, args, comm=comm)

    def run(self):
        """Run method

        Calls `Flow.posterior` to use the `Flow` object to get the posterior distribution

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Puts the data into the data store under this stages 'output' tag
        """

        data = self.get_data("input")
        flow = self.get_data("flow")
        if self.config.marg_rules is None:  # pragma: no cover
            marg_rules = {
                "flag": np.nan,
                "mag_u_lsst": lambda row: np.linspace(25, 31, 10),
            }
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

        ensemble = qp.Ensemble(
            qp.interp, data={"xvals": self.config.grid, "yvals": pdfs}
        )
        self.add_data("output", ensemble)
