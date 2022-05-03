import numpy as np
import pandas as pd

import tables_io
from ceci.config import StageParameter as Param
from rail.core.data import PqHandle
from rail.core.stage import RailStage

# external code
import hyperbolic


def_bands = ['u', 'g', 'r', 'i', 'z', 'y']
mag_cols = [f"mag_{band}_lsst" for band in def_bands]
magerr_cols = [f"mag_err_{band}_lsst" for band in def_bands]


class HyperbolicSmoothing(RailStage):
    """
    TODO...
    """

    name = 'HyperbolicSmoothing'
    config_options = RailStage.config_options.copy()
    config_options.update(
        zeropoint=Param(
            float, 0.0,
            msg="magnitude zeropoint"),
        magnitude_columns=Param(
            list, mag_cols,
            msg="magnitudes for which hyperbolic magnitudes are computed"),
        magerror_columns=Param(
            list, magerr_cols,
            msg="magnitude errors corresponding to magnitude_columns "
                "(assuming same ordering)")
    )
    inputs = [('input', PqHandle)]
    outputs = [('parameters', PqHandle)]

    def __init__(self, args, comm=None):
        super().__init__(args, comm)
        self.mag_names = self.config.magnitude_columns
        self.error_names = self.config.magerror_columns
        if (n_mag := len(self.mag_names)) != (n_err := len(self.error_names)):
            raise ValueError(
                "number of magnitude and magnitude error columns do not match "
                f"({n_mag} != {n_err})")
        self.ZP = self.config.zeropoint

    def run(self):
        # get input data
        data = self.get_data('input', allow_missing=True)
        fields = np.zeros(len(data), dtype=np.int)  # placeholder

        # compute the optimal smoothing factor b for each filter
        stats = []
        for mag_col, magerr_col in zip(self.mag_names, self.error_names):

            # compute fluxes from magnitudes
            mags = data[mag_col]
            magerrs = data[magerr_col]
            fluxes = np.exp((self.ZP - mags) / hyperbolic.pogson)
            fluxerrs = magerrs / hyperbolic.pogson * fluxes

            # compute the median flux error and zeropoint
            stats_filt = hyperbolic.compute_flux_stats(
                fluxes, fluxerrs, fields, zeropoint=self.ZP)
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
        stats = pd.concat(stats)
        self.add_data('parameters', stats)

    def compute(self, data):
        """...
        """
        self.set_data('input', data)
        self.run()
        self.finalize()
        return self.get_handle('parameters')


class HyperbolicMagnitudes(RailStage):
    """
    TODO...
    """

    name = 'HyperbolicMagnitudes'
    config_options = RailStage.config_options.copy()
    config_options.update(
        magnitude_columns=Param(
            list, mag_cols,
            msg="magnitudes for which hyperbolic magnitudes are computed"),
        magerror_columns=Param(
            list, magerr_cols,
            msg="magnitude errors corresponding to magnitude_columns "
                "(assuming same ordering)")
    )
    inputs = [('input', PqHandle),
              ('parameters', PqHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, comm=None):
        super().__init__(args, comm)
        self.mag_names = self.config.magnitude_columns
        self.error_names = self.config.magerror_columns
        if (n_mag := len(self.mag_names)) != (n_err := len(self.error_names)):
            raise ValueError(
                "number of magnitude and magnitude error columns do not match "
                f"({n_mag} != {n_err})")

    def run(self):
        # get input data
        data = self.get_data('input', allow_missing=True)
        stats = self.get_data('parameters', allow_missing=True)
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
            zeropoint = hyperbolic.fields_to_source(
                stats_filt[hyperbolic.Keys.zp], fields, index=data.index)

            # compute fluxes from magnitudes
            mags = data[mag_col].to_numpy()
            magerrs = data[magerr_col].to_numpy()
            fluxes = np.exp((zeropoint - mags) / hyperbolic.pogson)
            fluxerrs = magerrs / hyperbolic.pogson * fluxes

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

        # store result
        self.add_data('output', output)

    def compute(self, data, parameters):
        """...
        """
        self.set_data('input', data)
        self.set_data('parameters', parameters)
        self.run()
        self.finalize()
        return self.get_handle('output')
