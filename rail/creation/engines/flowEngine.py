"""This is the subclass of Creator that wraps a PZFlow Flow so that it can
be used to generate synthetic data and calculate posteriors."""

import numpy as np
import qp
from ceci.config import StageParameter as Param
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, RollingSplineCoupling, ShiftBounds
from rail.core.data import FlowHandle, PqHandle, QPHandle, TableHandle
from rail.creation.engine import Creator, Modeler, PosteriorCalculator


class FlowModeler(Modeler):
    """Modeler wrapper for a PZFlow Flow object.

    This class trains the flow.
    """

    name = "FlowModeler"
    inputs = [("input", TableHandle)]
    outputs = [("model", FlowHandle)]

    config_options = Modeler.config_options.copy()
    config_options.update(
        phys_cols=Param(
            dict,
            {"redshift": [0, 3]},
            msg="Names of non-photometry columns and their corresponding [min, max] values.",
        ),
        phot_cols=Param(
            dict,
            {
                "mag_u_lsst": [17, 35],
                "mag_g_lsst": [16, 32],
                "mag_r_lsst": [15, 30],
                "mag_i_lsst": [15, 30],
                "mag_z_lsst": [14, 29],
                "mag_y_lsst": [14, 28],
            },
            msg="Names of photometry columns and their corresponding [min, max] values.",
        ),
        calc_colors=Param(
            dict,
            {"ref_column_name": "mag_i_lsst"},
            msg=(
                "Whether to internally calculate colors (if phot_cols are magnitudes). "
                "Assumes that you want to calculate colors from adjacent columns in "
                "phot_cols. If you do not want to calculate colors, set False. Else, "
                "provide a dictionary {'ref_column_name': band}, where band is a string "
                "corresponding to the column in phot_cols you want to save as the "
                "overall galaxy magnitude."
            ),
        ),
        spline_knots=Param(
            int,
            16,
            msg="The number of spline knots in the normalizing flow.",
        ),
        num_training_epochs=Param(
            int,
            30,
            msg="The number of training epochs.",
        ),
        seed=Param(
            int,
            0,
            msg="The random seed for training.",
        ),
    )

    def __init__(self, args, comm=None):
        """Constructor

        Does standard Modeler initialization.
        """
        Modeler.__init__(self, args, comm=comm)

        # get the columns we are modeling
        phys_cols = self.config.phys_cols
        phot_cols = self.config.phot_cols

        # assemble the list of column names
        column_names = list(phys_cols) + list(phot_cols)

        # now let's set up the RQ-RSC
        nlayers = len(column_names)  # can make configurable from yaml if wanted
        transformed_dim = 1  # can make configurable from yaml if wanted
        K = self.config.spline_knots
        rsc = RollingSplineCoupling(nlayers, K=K, transformed_dim=transformed_dim)

        if color_config := self.config["calc_colors"]:
            # tell it which column to use as the reference magnitude
            ref_idx = column_names.index(color_config["ref_column_name"])
            # and which columns correspond to the magnitudes we want colors for
            mag_idx = [column_names.index(band) for band in phot_cols]

            # convert magnitude ranges to color ranges
            mag_ranges = np.array(list(phot_cols.values()))
            ref_mag_range = mag_ranges[
                list(phot_cols).index(color_config["ref_column_name"])
            ]
            color_ranges = mag_ranges[:-1] - mag_ranges[1:, ::-1]
            mins = (
                [col_range[0] for col_range in phys_cols.values()]
                + [ref_mag_range[0]]
                + list(color_ranges[:, 0])
            )
            maxs = (
                [col_range[1] for col_range in phys_cols.values()]
                + [ref_mag_range[1]]
                + list(color_ranges[:, 1])
            )

            # chain all the bijectors together
            bijector = Chain(
                ColorTransform(ref_idx, mag_idx),
                ShiftBounds(mins, maxs),
                rsc,
            )
        else:
            # get the list of mins and maxes
            mins = [
                col_range[0]
                for columns in [phys_cols, phot_cols]
                for col_range in columns.values()
            ]
            maxs = [
                col_range[1]
                for columns in [phys_cols, phot_cols]
                for col_range in columns.values()
            ]

            # chain the bijectors
            bijector = Chain(
                ShiftBounds(mins, maxs),
                rsc,
            )

        # build the flow
        self.flow = Flow(column_names, bijector=bijector)

    def run(self):
        """Run method

        Calls `Flow.train` to train a normalizing flow using PZFlow.

        Notes
        -----
        Puts the data into the data store under this stages 'output' tag
        """
        # get the catalog
        catalog = self.get_data("input")

        # train the flow
        _ = self.flow.train(
            catalog,
            epochs=self.config.num_training_epochs,
            verbose=True,
            seed=self.config.seed,
        )

        # save the flow
        self.add_data("model", self.flow)

        # TODO: NEED TO SAVE THE LOSSES SO WE CAN PLOT THEM


class FlowCreator(Creator):
    """Creator wrapper for a PZFlow Flow object."""

    name = "FlowCreator"
    inputs = [("model", FlowHandle)]
    outputs = [("output", PqHandle)]

    def __init__(self, args, comm=None):
        """Constructor

        Does standard Creator initialization and also gets the `Flow` object
        """
        Creator.__init__(self, args, comm=comm)

    def run(self):
        """Run method

        Calls `Flow.sample` to use the `Flow` object to generate photometric data

        Notes
        -----
        Puts the data into the data store under this stages 'output' tag
        """
        flow = self.get_data("model")
        if flow is None:  # pragma: no cover
            raise ValueError(
                "Tried to run a FlowCreator before the `Flow` model is loaded"
            )
        self.add_data(
            "output", flow.sample(self.config.n_samples, seed=self.config.seed)
        )


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
        {"flag": np.nan,
        "u": np.linspace(25, 31, 10),}
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

    inputs = [("model", FlowHandle), ("input", PqHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args, comm=None):
        """Constructor

        Does standard PosteriorCalculator initialization
        """
        PosteriorCalculator.__init__(self, args, comm=comm)

    def run(self):
        """Run method

        Calls `Flow.posterior` to use the `Flow` object to get the posterior
        distribution.

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Puts the data into the data store under this stages 'output' tag
        """
        # pull out the flow and the data we want posteriors for
        data = self.get_data("input")
        flow = self.get_data("model")

        # if no marginalization rules are set, use default values
        if self.config.marg_rules is None:  # pragma: no cover
            marg_rules = {
                "flag": np.nan,
                "mag_u_lsst": lambda row: np.linspace(25, 31, 10),
            }
        else:
            marg_rules = self.config.marg_rules

        # use the PZFlow normalizing flow to calculate posteriors
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

        # save the posteriors in a qp ensemble
        ensemble = qp.Ensemble(
            qp.interp, data={"xvals": self.config.grid, "yvals": pdfs}
        )

        self.add_data("output", ensemble)
