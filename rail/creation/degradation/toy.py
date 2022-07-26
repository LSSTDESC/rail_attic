""" LSST Model for photometric errors """

from numbers import Number
from typing import Iterable, List, Optional, Tuple

import numpy as np
from rail.creation.degradation import Degrader


class ToyModel(Degrader):
    """
    A toy degrader which only print out a given column

    Parameters
    ----------
    colname: string
    """

    name = 'toy'
    config_options = Degrader.config_options.copy()
    config_options.update(**{"colname": "u"})

    def __init__(self, args, comm=None):
        Degrader.__init__(self, args, comm=comm)

        # validate the settings
        self._validate_settings()


    def _validate_settings(self):
        """
        Validate all the settings.
        """

        # check that highSNR is boolean
        if isinstance(self.config["colname"], str) is not True:  #pragma: no cover
            raise TypeError("colname must be a string.")

    def run(self):
        """
        Run the toy
        """
        # get the bands and bandNames present in the data
        data = self.get_data('input', allow_missing=True)
        if self.config["colname"] is None:
            print(data)
        elif self.config["colname"] in data.columns:
            print(data[self.config["colname"]])
        else:
            raise ValueError("Column name "+self.config["colname"]+" not found!")

        self.add_data('output', data)



    def __repr__(self):  
        """
        Define how the model is represented and printed.
        """

        settings = self.config

        # start message
        printMsg = "Columns to print: " + self.config["colname"]

        return printMsg
