""" Stages that implement utility functions """

import pandas as pd

import tables_io
from rail.core.stage import RailStage

from rail.core.data import TableHandle

class ColumnMapper(RailStage):
    """Utility stage that remaps the names of columns.

    """
    name = 'ColumnMapper'
    config_options = dict(hdf5_groupname='', columns=dict, inplace=False)
    inputs = [('input', TableHandle)]
    outputs = [('output', TableHandle)]

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)

    def run(self):
        if self.config.hdf5_groupname:
            data = self.get_data('input')[self.config.hdf5_groupname]
        else:
            data = self.get_data('input')
        out_data = data.rename(columns=self.config.columns, inplace=self.config.inplace)
        if self.config.inplace:  #pragma: no cover
            out_data = data
        self.add_data('output', out_data)

    def __repr__(self):  # pragma: no cover
        printMsg = "Stage that applies remaps the following column names in a pandas DataFrame:\n"
        printMsg += "f{str(self.config.columns)}"
        return printMsg

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a table with the columns names changed

        Parameters
        ----------
        sample : pd.DataFrame
            The data to be renamed

        Returns
        -------
        pd.DataFrame
            The degraded sample
        """
        self.set_data('input', data)
        self.run()
        return self.get_handle('output')


class RowSelector(RailStage):
    """Utility stage that remaps the names of columns.

    """
    name = 'RowSelector'
    config_options = dict(hdf5_groupname='', start=int, stop=int)
    inputs = [('input', TableHandle)]
    outputs = [('output', TableHandle)]

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)

    def run(self):
        if self.config.hdf5_groupname:
            data = self.get_data('input')[self.config.hdf5_groupname]
        else:
            data = self.get_data('input')
        out_data = data.iloc[3:4]
        self.add_data('output', out_data)

    def __repr__(self):  # pragma: no cover
        printMsg = "Stage that applies remaps the following column names in a pandas DataFrame:\n"
        printMsg += "f{str(self.config.columns)}"
        return printMsg

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a table with the columns names changed

        Parameters
        ----------
        sample : pd.DataFrame
            The data to be renamed

        Returns
        -------
        pd.DataFrame
            The degraded sample
        """
        self.set_data('input', data)
        self.run()
        return self.get_handle('output')


class TableConverter(RailStage):
    """Utility stage that converts tables from one format to another"""
    name = 'TableConverter'
    config_options = dict(hdf5_groupname='', output_format=str)
    inputs = [('input', TableHandle)]
    outputs = [('output', TableHandle)]

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)

    def run(self):
        if self.config.hdf5_groupname:
            data = self.get_data('input')[self.config.hdf5_groupname]
        else:
            data = self.get_data('input')
        out_fmt = tables_io.types.TABULAR_FORMAT_NAMES[self.config.output_format]
        out_data = tables_io.convert(data, out_fmt)
        self.add_data('output', out_data)

    def __call__(self, data):
        """Return a converted table

        Parameters
        ----------
        data : table-like
            The data to be converted

        Returns
        -------
        out_data : table-like
            The converted version of the table
        """
        self.set_data('input', data)
        self.run()
        return self.get_handle('output')
