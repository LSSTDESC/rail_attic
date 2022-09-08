"""Degrader that applies a cut to given columns."""

from numbers import Number

import numpy as np
from rail.creation.degrader import Degrader


class QuantityCut(Degrader):
    """Degrader that applies a cut to the given columns.

    Note if a galaxy fails any of the cuts on any one of its columns, that
    galaxy is removed from the sample.
    """

    name = "QuantityCut"
    config_options = Degrader.config_options.copy()
    config_options.update(cuts=dict)

    def __init__(self, args, comm=None):
        """
        Constructor

        Does standard Degrader initialization and also gets defines the cuts to be applied
        """
        Degrader.__init__(self, args, comm=comm)
        self.cuts = None
        self.set_cuts(self.config["cuts"])

    def set_cuts(self, cuts: dict):
        """
        Parameters
        ----------
        cuts : dict
            A dictionary of cuts to make on the data.

        Notes
        -----
        The cut keys should be the names of columns you wish to make cuts on.
        The cut values should be either:
        - a number, which is the maximum value. I.e. if the dictionary
        contains "i": 25, then values of i > 25 are cut from the sample.
        - an iterable, which is the range of acceptable values.
        I.e. if the dictionary contains "redshift": (1.5, 2.3), then
        redshifts outside that range are cut from the sample.
        """

        # check that cuts is a dictionary
        if not isinstance(cuts, dict):  # pragma: no cover
            raise TypeError("cuts must be a dictionary.")

        # validate all the cuts and standardize format in dictionary
        self.cuts = {}
        for quantity, cut in cuts.items():
            bad_cut_msg = (
                f"Cut for {quantity} must be a number or an iterable of (min, max)"
            )
            # if single number is provided, save that as the maximum
            if isinstance(cut, Number):
                self.cuts[quantity] = (-np.inf, cut)
            # else, if it's an iterable...
            elif hasattr(cut, "__iter__"):
                # if the iterable is a string or dict, raise error
                if isinstance(cut, (dict, str)):
                    raise TypeError(bad_cut_msg)
                # if the length of the iterable isn't 2, raise error
                if len(cut) != 2:
                    raise ValueError(bad_cut_msg)
                # check that both of the cut values are a number
                for c in cut:
                    if not isinstance(c, Number):
                        raise TypeError(bad_cut_msg)
                # check that the cuts are (min, max)
                if cut[0] > cut[1]:
                    raise ValueError(
                        f"The max for {quantity} must be greater than the min."
                    )
                # if all of these checks passed, save the cuts
                self.cuts[quantity] = (cut[0], cut[1])
            # if the cut isn't a number or iterable, it's the wrong type
            else:
                raise TypeError(bad_cut_msg)

    def run(self):
        """Run method

        Applies cuts

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Puts the data into the data store under this stages 'output' tag
        """
        data = self.get_data("input")

        # get overlap of columns from data and columns on which to make cuts
        columns = set(self.cuts.keys()).intersection(data.columns)

        if len(columns) == 0:  # pragma: no cover
            self.add_data("output", data)
        else:
            # generate a pandas query from the cuts
            query = [
                f"{col} > {self.cuts[col][0]} & {col} < {self.cuts[col][1]}"
                for col in columns
            ]
            query = " & ".join(query)

            out_data = data.query(query)
            self.add_data("output", out_data)

    def __repr__(self):  # pragma: no cover
        """Pretty print this object"""
        printMsg = "Degrader that applies the following cuts to a pandas DataFrame:\n"
        printMsg += "{column: (min, max), ...}\n"
        printMsg += self.cuts.__str__()
        return printMsg
