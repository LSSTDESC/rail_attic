""" Stages that implement utility functions """
import os
import numpy as np
import pandas as pd

import tables_io
from astropy.coordinates import SkyCoord

from rail.core.stage import RailStage

from rail.core.data import PqHandle, Hdf5Handle

dustmaps_config = tables_io.lazy_modules.lazyImport('dustmaps.config')
dustmaps_sfd = tables_io.lazy_modules.lazyImport('dustmaps.sfd')


class ColumnMapper(RailStage):
    """Utility stage that remaps the names of columns.

    Notes
    -----
    1. This operates on pandas dataframs in parquet files.

    2. In short, this does:
    `output_data = input_data.rename(columns=self.config.columns, inplace=self.config.inplace)`

    """
    name = 'ColumnMapper'
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=100_000, columns=dict, inplace=False)
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)

    def run(self):
        data = self.get_data('input', allow_missing=True)
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
    """Utility Stage that sub-selects rows from a table by index

    Notes
    -----
    1. This operates on pandas dataframs in parquet files.

    2. In short, this does:
    `output_data = input_data[self.config.start:self.config.stop]`

    """
    name = 'RowSelector'
    config_options = RailStage.config_options.copy()
    config_options.update(start=int, stop=int)
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)

    def run(self):
        data = self.get_data('input', allow_missing=True)
        out_data = data.iloc[self.config.start:self.config.stop]
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


class LSSTFluxToMagConverter(RailStage):
    """Utility stage that converts from fluxes to magnitudes

    Note, this is hardwired to take parquet files as input
    and provide hdf5 files as output
    """
    name = 'LSSTFluxToMagConverter'

    config_options = RailStage.config_options.copy()
    config_options.update(bands='ugrizy')
    config_options.update(flux_name="{band}_gaap1p0Flux")
    config_options.update(flux_err_name="{band}_gaap1p0FluxErr")
    config_options.update(mag_name="mag_{band}_lsst")
    config_options.update(mag_err_name="mag_err_{band}_lsst")
    config_options.update(copy_cols={})
    config_options.update(mag_offset=31.4)

    mag_conv = np.log(10)*0.4

    inputs = [('input', PqHandle)]
    outputs = [('output', Hdf5Handle)]

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)

    def _flux_to_mag(self, flux_vals):
        return -2.5*np.log10(flux_vals) + self.config.mag_offset

    def _flux_err_to_mag_err(self, flux_vals, flux_err_vals):
        return flux_err_vals / (flux_vals*self.mag_conv)

    def run(self):
        data = self.get_data('input', allow_missing=True)
        out_data = {}
        const = np.log(10.)*0.4
        for band_ in self.config.bands:
            flux_col_name = self.config.flux_name.format(band=band_)
            flux_err_col_name = self.config.flux_err_name.format(band=band_)
            out_data[self.config.mag_name.format(band=band_)] = self._flux_to_mag(data[flux_col_name].values)
            out_data[self.config.mag_err_name.format(band=band_)] = self._flux_err_to_mag_err(data[flux_col_name].values, data[flux_err_col_name].values)
        for key, val in self.config.copy_cols.items():  # pragma: no cover
            out_data[key] = data[val].values
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


class TableConverter(RailStage):
    """Utility stage that converts tables from one format to anothe

    FIXME, this is hardwired to convert parquet tables to Hdf5Tables.
    It would be nice to have more options here.
    """
    name = 'TableConverter'
    config_options = RailStage.config_options.copy()
    config_options.update(output_format=str)
    inputs = [('input', PqHandle)]
    outputs = [('output', Hdf5Handle)]

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)

    def run(self):
        data = self.get_data('input', allow_missing=True)
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


class Dereddener(RailStage):
    """Utility stage that does dereddening

    """
    name = 'Dereddener'

    config_options = RailStage.config_options.copy()
    config_options.update(bands='ugrizy')
    config_options.update(mag_name="mag_{band}_lsst")
    config_options.update(band_a_env=[4.81,3.64,2.70,2.06,1.58,1.31])
    config_options.update(dustmap_name='sfd')
    config_options.update(dustmap_dir=str)
    config_options.update(copy_cols=[])

    inputs = [('input', Hdf5Handle)]
    outputs = [('output', Hdf5Handle)]

    def fetch_map(self):
        dust_map_dict = dict(sfd=dustmaps_sfd)
        try:
            dust_map_submod = dust_map_dict[self.config.dustmap_name]
        except KeyError as msg:  # pragma: no cover
            raise KeyError(f"Unknown dustmap {self.config.dustmap_name}, options are {list(dust_map_dict.keys())}") from msg

        if os.path.exists(os.path.join(self.config.dustmap_dir, self.config.dustmap_name)):  # pragma: no cover
            # already downloaded, return
            return
        
        dust_map_config = dustmaps_config.config
        dust_map_config['data_dir'] = self.config.dustmap_dir
        fetch_func = dust_map_submod.fetch
        fetch_func()
        
            
    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)

    def run(self):
        data = self.get_data('input', allow_missing=True)
        out_data = {}
        coords = SkyCoord(data['ra'], data['decl'], unit = 'deg',frame='fk5')
        dust_map_dict = dict(sfd=dustmaps_sfd.SFDQuery)
        try:
            dust_map_class = dust_map_dict[self.config.dustmap_name]
            dust_map_config = dustmaps_config.config
            dust_map_config['data_dir'] = self.config.dustmap_dir
            dust_map = dust_map_class()
        except KeyError as msg:  # pragma: no cover
            raise KeyError(f"Unknown dustmap {self.config.dustmap_name}, options are {list(dust_map_dict.keys())}") from msg
        ebvvec = dust_map(coords)
        for i, band_ in enumerate(self.config.bands):
            band_mag_name = self.config.mag_name.format(band=band_)
            mag_vals = data[band_mag_name]
            out_data[band_mag_name] = mag_vals - ebvvec*self.config.band_a_env[i]
        for col_ in self.config.copy_cols:  # pragma: no cover
            out_data[col_] = data[col_]
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
