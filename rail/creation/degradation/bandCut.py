import numpy as np
import pandas as pd
from rail.creation.degradation import Degrader


class BandCut(Degrader):
    """Degrader that applies a cut to the given columns."""

    def __init__(self, cuts: dict):
        """
        Parameters
        ----------
        cuts : dict
            A dictionary of cuts to make on the data.
            They keys should be the names of columns you wish to make cuts on.
            The values should be either:
                - a number, which is the maximum value. I.e. if the dictionary
                contains "i": 25, then values of i > 25 are cut from the sample.
                - an iterable, which is the range of acceptable values.
                I.e. if the dictionary contains "redshift": (1.5, 2.3), then
                redshifts outside that range are cut from the sample.
        """

        # check that cuts is a dictionary
        assert isinstance(cuts, dict), "cuts must be a dictionary."

        # validate all the cuts and standardize format in dictionary
        self.cuts = dict()
        for band, cut in cuts.items():
            bad_cut_msg = (
                f"Cut for {band} must be a number or an iterable of (min, max)"
            )
            if isinstance(cut, float) or isinstance(cut, int):
                self.cuts[band] = (-np.inf, cut)
            elif hasattr(cut, "__iter__") and not isinstance(cut, str):
                assert len(cut) == 2, bad_cut_msg
                for c in cut:
                    assert isinstance(c, float) or isinstance(c, int), bad_cut_msg
                self.cuts[band] = (cut[0], cut[1])
            else:
                raise ValueError(bad_cut_msg)

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:

        # get overlap of columns from data and columns on which to make cuts
        columns = set(self.cuts.keys()).intersection(data.columns)

        # generate a pandas query from the cuts
        query = [
            f"{col} > {self.cuts[col][0]} & {col} < {self.cuts[col][1]}"
            for col in columns
        ]
        query = " & ".join(query)

        return data.query(query)

    def __repr__(self):
        printMsg = "Degrader that applies the following cuts to a pandas DataFrame:\n"
        printMsg += "{column: (min, max), ...}\n"
        printMsg += self.cuts.__str__()
        return printMsg
