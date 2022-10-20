"""
Module that implements operations on photometric data such as magnitudes and fluxes.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ceci.config import StageParameter as Param
from rail.core.data import PqHandle
from rail.core.stage import RailStage

import hyperbolic  # https://github.com/jlvdb/hyperbolic


# default column names in DC2
LSST_BANDS = 'ugrizy'
DEFAULT_MAG_COLS = [f"mag_{band}_lsst" for band in LSST_BANDS]
DEFAULT_MAGERR_COLS = [f"mag_err_{band}_lsst" for band in LSST_BANDS]


def _compute_flux(magnitude, zeropoint):
    """
    Compute the flux corresponding to a given magnitude and photometric zeropoint.

    Parameters
    ----------
    magnitude : array-like
        Magnitude or array of magnitudes.
    zeropoint : array-like
        Photometric zeropoint used for conversion.

    Returns
    -------
    flux : array-like
        Flux value(s).
    """
    flux = np.exp((zeropoint - magnitude) / hyperbolic.pogson)
    return flux


def _compute_flux_error(flux, magnitude_error):
    """
    Compute the flux error corresponding to a given flux and magnitude error.

    Parameters
    ----------
    flux : array-like
        Flux or array of fluxes.
    magnitude_error : array-like
        Magnitude error or array of magnitude errors.

    Returns
    -------
    flux_error : array-like
        Flux error value(s).
    """
    flux_error = magnitude_error / hyperbolic.pogson * flux
    return flux_error


class PhotormetryManipulator(RailStage, ABC):
    """
    Base class to perform opertations on magnitudes. A table with input magnitudes and errors is
    processed and transformed into an output table with new magnitudes and errors.

    Subclasses must implement the run() and compute() method.
    """

    name = 'PhotormetryManipulator'
    config_options = RailStage.config_options.copy()
    config_options.update(
        value_columns=Param(
            list, default=DEFAULT_MAG_COLS,
            msg="list of columns that prove photometric measurements (fluxes or magnitudes)"),
        error_columns=Param(
            list, default=DEFAULT_MAGERR_COLS,
            msg="list of columns with errors corresponding to value_columns "
                "(assuming same ordering)"),
        zeropoints=Param(
            list, default=[], required=False,
            msg="optional list of magnitude zeropoints for value_columns "
                "(assuming same ordering, defaults to 0.0)"),
        is_flux=Param(
            bool, default=False,
            msg="whether the provided quantities are fluxes or magnitudes"))
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, comm=None):
        super().__init__(args, comm)
        self._check_config()
        # convenience remapping of parameters
        self.value_columns = self.config.value_columns
        self.error_columns = self.config.error_columns
        self.zeropoints = self.config.zeropoints
        self.n_col = len(self.value_columns)

    def _check_config(self):
        # compare column definitions
        n_mag = len(self.config.value_columns)
        n_err = len(self.config.error_columns)
        n_zpt = len(self.config.zeropoints)
        if n_mag != n_err:
            raise IndexError(
                f"number of magnitude and error columns do not match ({n_mag} != {n_err})")
        # check and zeropoints or parse default value
        if n_zpt == 0:
            self.config.zeropoints = [0.0] * n_mag
        elif n_zpt != n_mag:
            raise IndexError(
                f"number of zeropoints and magnitude columns do not match ({n_zpt} != {n_mag})")

    def get_as_fluxes(self):
        """
        Loads specified photometric data as fluxes, converting magnitudes on the fly.
        """
        input_data = self.get_data('input', allow_missing=True)
        if self.config.is_flux:
            data = input_data[self.value_columns + self.error_columns]
        else:
            data = pd.DataFrame()
            # convert magnitudes to fluxes
            for val_name, zeropoint in zip(self.value_columns, self.zeropoints):
                data[val_name] = _compute_flux(
                    input_data[val_name],
                    zeropoint=zeropoint)
            # compute flux errors from magnitude errors
            for val_name, err_name in zip(self.value_columns, self.error_columns):
                data[err_name] = _compute_flux_error(
                    data[val_name],
                    input_data[err_name])
        return data

    @abstractmethod
    def run(self):  # pragma: no cover
        """
        Implements the operation performed on the photometric data.
        """
        data = self.get_as_fluxes()
        # do work
        self.add_data('output', data)

    @abstractmethod
    def compute(self, data):  # pragma: no cover
        """
        Main method to call.

        Parameters
        ----------
        data : `PqHandle`
           Input tabular data with column names as defined in the configuration.

        Returns
        -------
        output: `PqHandle`
            Output tabular data.
        """
        self.set_data('input', data)
        self.run()
        self.finalize()
        return self.get_handle('output')


class HyperbolicSmoothing(PhotormetryManipulator):
    """
    Initial stage to compute hyperbolic magnitudes (Lupton et al. 1999). Estimates the smoothing
    parameter b that is used by the second stage (`HyperbolicMagnitudes`) to convert classical to
    hyperbolic magnitudes.
    """

    name = 'HyperbolicSmoothing'
    config_options = PhotormetryManipulator.config_options.copy()
    inputs = [('input', PqHandle)]
    outputs = [('parameters', PqHandle)]

    def run(self):
        """
        Computes the smoothing parameter b (see Lupton et al. 1999) per photometric band.
        """
        # get input data
        data = self.get_as_fluxes()
        fields = np.zeros(len(data), dtype=int)  # placeholder

        # compute the optimal smoothing factor b for each photometric band
        stats = []
        for fx_col, fxerr_col, zeropoint in zip(
                self.value_columns, self.error_columns, self.zeropoints):

            # compute the median flux error and zeropoint
            stats_filt = hyperbolic.compute_flux_stats(
                data[fx_col], data[fxerr_col], fields, zeropoint=zeropoint)
            # compute the smoothing parameter b (in normalised flux)
            stats_filt[hyperbolic.Keys.b] = hyperbolic.estimate_b(
                stats_filt[hyperbolic.Keys.zp],
                stats_filt[hyperbolic.Keys.flux_err])
            # compute the smoothing parameter b (in absolute flux)
            stats_filt[hyperbolic.Keys.b_abs] = (
                stats_filt[hyperbolic.Keys.ref_flux] *
                stats_filt[hyperbolic.Keys.b])

            # collect results
            stats_filt[hyperbolic.Keys.filter] = fx_col
            stats_filt = stats_filt.reset_index().set_index([
                hyperbolic.Keys.filter,
                hyperbolic.Keys.field])
            stats.append(stats_filt)

        # store resulting smoothing parameters for next stage
        self.add_data('parameters', pd.concat(stats))

    def compute(self, data):
        """
        Main method to call. Computes the set of smoothing parameters (b) for an input catalogue
        with classical photometry and their respective errors. These parameters are required by the
        follow-up stage `HyperbolicMagnitudes` and are parsed as tabular data.

        Parameters
        ----------
        data : `PqHandle`
            Input table with magnitude and magnitude error columns as defined in the configuration.

        Returns
        -------
        parameters : `PqHandle`
            Table with smoothing parameters per photometric band and additional meta data.
        """
        self.set_data('input', data)
        self.run()
        self.finalize()
        return self.get_handle('parameters')


class HyperbolicMagnitudes(PhotormetryManipulator):
    """
    Convert a set of classical magnitudes to hyperbolic magnitudes  (Lupton et al. 1999). Requires
    input from the initial stage (`HyperbolicSmoothing`) to supply optimal values for the smoothing
    parameters (b).
    """

    name = 'HyperbolicMagnitudes'
    config_options = PhotormetryManipulator.config_options.copy()
    inputs = [('input', PqHandle),
              ('parameters', PqHandle)]
    outputs = [('output', PqHandle)]

    def _check_filters(self, stats_table):
        """
        Check whether the column definition matches the loaded smoothing parameters.

        Parameters:
        -----------
        stats_table : `pd.DataFrame`
            Data table that contains smoothing parameters per photometric band (from
            `HyperbolicSmoothing`).

        Raises
        ------
        KeyError : Filter defined in magnitude_columns is not found in smoothing parameter table.
        """
        # filters in the parameter table
        param_filters = set(stats_table.reset_index()[hyperbolic.Keys.filter])
        # filters parsed through configuration
        config_filters = set(self.value_columns)
        # check if the filters match
        filter_diff = config_filters - param_filters
        if len(filter_diff) != 0:
            strdiff = ", ".join(sorted(filter_diff))
            raise KeyError(f"parameter table contains no smoothing parameters for: {strdiff}")

    def run(self):
        """
        Compute hyperbolic magnitudes and their error based on the parameters determined by
        `HyperbolicSmoothing`.
        """
        # get input data
        data = self.get_as_fluxes()
        stats = self.get_data('parameters', allow_missing=True)
        self._check_filters(stats)
        fields = np.zeros(len(data), dtype=int)  # placeholder for variable field/pointing depth

        # intialise the output data
        output = pd.DataFrame(index=data.index)  # allows joining on input

        # compute smoothing parameter b
        b = stats[hyperbolic.Keys.b].groupby(  # median flux error in each filter
            hyperbolic.Keys.filter).agg(np.nanmedian)
        b = b.to_dict()

        # hyperbolic magnitudes
        for val_col, err_col in zip(self.value_columns, self.error_columns):
            # get the smoothing parameters
            stats_filt = hyperbolic.fill_missing_stats(stats.loc[val_col])

            # map reference flux from fields/pointings to sources
            ref_flux_per_source = hyperbolic.fields_to_source(
                stats_filt[hyperbolic.Keys.ref_flux], fields, index=data.index)
            norm_flux = data[val_col] / ref_flux_per_source
            norm_flux_err = data[err_col] / ref_flux_per_source

            # compute the hyperbolic magnitudes
            hyp_mag = hyperbolic.compute_magnitude(
                norm_flux, b[val_col])
            hyp_mag_err = hyperbolic.compute_magnitude_error(
                norm_flux, b[val_col], norm_flux_err)

            # add data to catalogue
            key_mag = val_col.replace("mag_", "mag_hyp_")
            key_mag_err = err_col.replace("mag_", "mag_hyp_")
            output[key_mag] = hyp_mag
            output[key_mag_err] = hyp_mag_err

        # store results
        self.add_data('output', output)

    def compute(self, data, parameters):
        """
        Main method to call. Outputs hyperbolic magnitudes compuated from a set of smoothing
        parameters and input catalogue with classical magitudes and their respective errors.

        Parameters
        ----------
        data : `PqHandle`
            Input table with photometry (magnitudes or flux columns and their respective
            uncertainties) as defined by the configuration.
        parameters : `PqHandle`
            Table with smoothing parameters per photometric band, determined by
            `HyperbolicSmoothing`.

        Returns
        -------
        output: `PqHandle`
            Output table containting hyperbolic magnitudes and their uncertainties. If the columns
            in the input table contain a prefix `mag_`, this output tabel will replace the prefix
            with `hyp_mag_`, otherwise the column names will be identical to the input table.
        """
        self.set_data('input', data)
        self.set_data('parameters', parameters)
        self.run()
        self.finalize()
        return self.get_handle('output')
