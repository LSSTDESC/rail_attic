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
lsst_bands = 'ugrizy'
mag_cols = [f"mag_{band}_lsst" for band in lsst_bands]
magerr_cols = [f"mag_err_{band}_lsst" for band in lsst_bands]


def _compute_flux(mags, zeropoint):
    """
    Compute the flux corresponding to a given magnitude and photometric zeropoint.

    Parameters
    ----------
    mags : array-like
        Magnitude or array of magnitudes.
    zeropoint : `float`
        Photometric zeropoint used for conversion.

    Returns
    -------
    fluxes : array-like
        Flux values.
    """
    fluxes = np.exp((zeropoint - mags) / hyperbolic.pogson)
    return fluxes


def _compute_flux_error(fluxes, magerrs):
    """
    Compute the flux error corresponding to a given flux and magnitude error.

    Parameters
    ----------
    fluxes : array-like
        Flux or array of flux values.
    magerrs : array-like
        Magnitude error or array of magnitude errors.

    Returns
    -------
    flux_errors : array-like
        Flux error values.
    """
    flux_errors = magerrs / hyperbolic.pogson * fluxes
    return flux_errors


class MagnitudeManipulator(RailStage, ABC):
    """
    Base class to perform opertations on magnitudes. A table with input magnitudes and errors is
    processed and transformed into an output table with magnitudes and errors.

    Subclasses must implement the run() and compute() method.
    """

    name = 'MagnitudeManipulator'
    config_options = RailStage.config_options.copy()
    config_options.update(
        magnitude_columns=Param(
            list, mag_cols,
            msg="magnitudes for which hyperbolic magnitudes are computed"),
        magerror_columns=Param(
            list, magerr_cols,
            msg="magnitude errors corresponding to magnitude_columns (assuming same ordering)"))
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, comm=None):
        super().__init__(args, comm)
        self.mag_names = self.config.magnitude_columns
        self.error_names = self.config.magerror_columns
        if (n_mag := len(self.mag_names)) != (n_err := len(self.error_names)):
            raise ValueError(
                f"number of magnitude and error columns do not match ({n_mag} != {n_err})")

    @abstractmethod
    def run(self):
        """
        Implements the operation performed on the input data.
        """
        data = self.get_data('input', allow_missing=True)
        self.add_data('parameters', data)

    @abstractmethod
    def compute(self, data):
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


class HyperbolicSmoothing(MagnitudeManipulator):
    """
    Initial stage to compute hyperbolic magnitudes (Lupton et al. 1999). Estimates the smoothing
    parameter b that is used by the second stage (`HyperbolicMagnitudes`) to convert classical to
    hyperbolic magnitudes.
    """

    name = 'HyperbolicSmoothing'
    config_options = MagnitudeManipulator.config_options.copy()
    config_options.update(
        zeropoint=Param(
            float, 0.0,
            msg="magnitude zeropoint"))
    inputs = [('input', PqHandle)]
    outputs = [('parameters', PqHandle)]

    def __init__(self, args, comm=None):
        super().__init__(args, comm)
        self.zeropoint = self.config.zeropoint

    def run(self):
        """
        Computes the smoothing parameter b (see Lupton et al. 1999) per photometric band.
        """
        # get input data
        data = self.get_data('input', allow_missing=True)
        fields = np.zeros(len(data), dtype=np.int)  # placeholder

        # compute the optimal smoothing factor b for each photometric band
        stats = []
        for mag_col, magerr_col in zip(self.mag_names, self.error_names):

            # compute fluxes from magnitudes
            fluxes = _compute_flux(data[mag_col], self.zeropoint)
            fluxerrs = _compute_flux_error(fluxes, data[magerr_col])

            # compute the median flux error and zeropoint
            stats_filt = hyperbolic.compute_flux_stats(
                fluxes, fluxerrs, fields, zeropoint=self.zeropoint)
            # compute the smoothing parameter b (in normalised flux)
            stats_filt[hyperbolic.Keys.b] = hyperbolic.estimate_b(
                stats_filt[hyperbolic.Keys.zp],
                stats_filt[hyperbolic.Keys.flux_err])
            # compute the smoothing parameter b (in absolute flux)
            stats_filt[hyperbolic.Keys.b_abs] = (
                stats_filt[hyperbolic.Keys.ref_flux] *
                stats_filt[hyperbolic.Keys.b])

            # collect results
            stats_filt[hyperbolic.Keys.filter] = mag_col
            stats_filt = stats_filt.reset_index().set_index([
                hyperbolic.Keys.filter,
                hyperbolic.Keys.field])
            stats.append(stats_filt)

        # store resulting smoothing parameters for next stage
        self.add_data('parameters', pd.concat(stats))

    def compute(self, data):
        """
        Main method to call. Computes the set of smoothing parameters for an input catalogue with
        classical magitudes and their respective errors. These parameters are required by the
        follow-up stage `HyperbolicMagnitudes`.

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


class HyperbolicMagnitudes(MagnitudeManipulator):
    """
    Convert a set of classical magnitudes to hyperbolic magnitudes  (Lupton et al. 1999). Requires
    input from the initial stage (`HyperbolicSmoothing`) to supply optimal values for the smoothing
    parameter.
    """

    name = 'HyperbolicMagnitudes'
    config_options = MagnitudeManipulator.config_options.copy()
    inputs = [('input', PqHandle),
              ('parameters', PqHandle)]
    outputs = [('output', PqHandle)]

    def _check_filters(self, stats_table):
        """
        Check whether the column definition matche the loaded smoothing parameters.

        Parameters:
        -----------
        stats_table : `pd.DataFrame`
            Data table that contains smoothing parameters per photometric band (from
            `HyperbolicSmoothing`).

        Raises
        ------
        ValueError : Filter defined in magnitude_columns is not found in smoothing parameter table.
        """
        param_filters = set(stats_table.reset_index()[hyperbolic.Keys.filter])
        config_filters = set(self.mag_names)
        filter_diff = config_filters - param_filters
        if len(filter_diff) != 0:
            strdiff = ", ".join(sorted(filter_diff))
            raise ValueError(f"parameter table contains no smoothing parameters for: {strdiff}")

    def run(self):
        """
        Compute hyperbolic magnitudes based on the parameters determined by `HyperbolicSmoothing`.
        """
        # get input data
        data = self.get_data('input', allow_missing=True)
        stats = self.get_data('parameters', allow_missing=True)
        self._check_filters(stats)
        fields = np.zeros(len(data), dtype=np.int)  # placeholder

        # intialise the output data
        output = pd.DataFrame(index=data.index)  # allows joining on input

        # compute smoothing parameter b and zeropoint
        b = stats[hyperbolic.Keys.b].groupby(
            hyperbolic.Keys.filter).agg(np.nanmedian)
        b = b.to_dict()

        # compute flux errors
        for mag_col, magerr_col in zip(self.mag_names, self.error_names):
            # get the smoothing parameters
            stats_filt = hyperbolic.fill_missing_stats(stats.loc[mag_col])

            # get the zeropoint per source
            zeropoints = hyperbolic.fields_to_source(
                stats_filt[hyperbolic.Keys.zp], fields, index=data.index)

            # compute fluxes from magnitudes
            fluxes = _compute_flux(data[mag_col], zeropoints)
            fluxerrs = _compute_flux_error(fluxes, data[magerr_col])

            # compute normalised flux
            ref_flux = hyperbolic.fields_to_source(
                stats_filt[hyperbolic.Keys.ref_flux], fields, index=data.index)
            norm_flux = fluxes / ref_flux
            norm_flux_err = fluxerrs / ref_flux

            # compute the hyperbolic magnitudes
            hyp_mag = hyperbolic.compute_magnitude(
                norm_flux, b[mag_col])
            hyp_mag_err = hyperbolic.compute_magnitude_error(
                norm_flux, b[mag_col], norm_flux_err)

            # add data to catalogue
            key_mag = mag_col.replace("mag_", "mag_hyp_")
            key_mag_err = magerr_col.replace("mag_", "mag_hyp_")
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
            Input table with magnitude and magnitude error columns as defined in the configuration.
        parameters : `PqHandle`
            Table with smoothing parameters per photometric band, determined by
            `HyperbolicSmoothing`.

        Returns
        -------
        output: `PqHandle`
            Output table containting the values for the smoothing parameter that define the flux
            scale at which hyperbolic magnitdes transition from linear to logarithmic behaviour.
        """
        self.set_data('input', data)
        self.set_data('parameters', parameters)
        self.run()
        self.finalize()
        return self.get_handle('output')
