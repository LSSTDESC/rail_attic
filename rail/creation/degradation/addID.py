"""Degrader that adds a galaxyID."""

import numpy as np
from rail.creation.degradation import Degrader


class AddID(Degrader):
    """Degrader that simply adds a galaxyID to data.

    """
    name = 'AddID'
    config_options = Degrader.config_options.copy()

    def __init__(self, args, comm=None):
        """
        Constructor

        Does standard Degrader initialization and also gets defines the cuts to be applied
        """
        Degrader.__init__(self, args, comm=comm)

    def run(self):
        """ Run method

            Adds galaxy_id

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Puts the data into the data store under this stages 'output' tag
        """
        data = self.get_data('input')

        if 'id' not in data.keys():
            exkey = data.keys()[0]
            numgals = len(data[exkey])
            data['id'] = np.arange(numgals)
        
        self.add_data('output', data)

    def __repr__(self):  # pragma: no cover
        """ Pretty print this object """
        printMsg = "Degrader that adds an id column to pandas DataFrame:\n"
        return printMsg
