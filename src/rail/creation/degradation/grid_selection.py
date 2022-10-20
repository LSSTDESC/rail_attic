"""Degraders that emulate spectroscopic effects on photometry"""

import os
import numpy as np
import pandas as pd
import pickle
import tables_io
from rail.creation.degrader import Degrader
from rail.core.utils import RAILDIR
from ceci.config import StageParameter as Param


class GridSelection(Degrader):
    """
    Uses the ratio of HSC spectroscpic galaxies to photometric galaxies to portion a sample
    into training and application samples. Option to implement a color-based redshift cut off in each pixel.
    Option of further degrading the training sample by limiting it to galaxies less than a redshift cutoff by specifying redshift_cut.

    Parameters
    ----------
    color_redshift_cut: True or false, implements color-based redshift cut. Default is True.
        If True, ratio_file must include second key called 'data' with magnitudes, colors and spec-z from the spectroscopic sample.
    percentile_cut: If using color-based redshift cut, percentile in spec-z above which redshifts will be cut from training sample. Default is 99.0
    scaling_factor: Enables the user to adjust the ratios by this factor to change the overall number of galaxies kept.  For example, if you wish
        to generate 100,00 galaxies but only 50,000 are selected by default, then you can adjust factor up by a factor of 2 to return more galaixes.
    redshift_cut: redshift above which all galaxies will be removed from training sample. Default is 100
    ratio_file: hdf5 file containing an array of spectroscpic vs. photometric galaxies in each pixel. Default is hsc_ratios.hdf5 for an HSC based selection
    settings_file: pickled dictionary containing information about colors and magnitudes used in defining the pixels. Dictionary must include the following keys:
      'x_band_1': string, this is the band used for the magnitude in the color magnitude diagram. Default for HSC is 'i'.
      'x_band_2': string, this is the redder band used for the color in the color magnitude diagram.
      if x_band_2 string is not set to '' then the grid is assumed to be over color and x axis color is set to x_band_1 - x_band_2, default is ''.
      'y_band_1': string, this is the bluer band used for the color in the color magnitude grid. Default for HSC is 'g'.
      'y_band_2': string, this is the redder band used for the color in the color magnitude diagram.
      if y_band_2 is not set to '' then the y-band is assumed to be over color and is set to y_band_1 - y_band 2.
      'x_limits': 2-element list, this is a list of the lower and upper limits of the magnitude. Default for HSC is [13, 16],
      'y_limits': 2-element list, this is a list of the lower and upper limits of the color. Default for HSC is [-2, 6]}

    NOTE: the default 'HSC' grid file, located in RAIL/examples/creation/data/hsc_ratios_and_specz.hdf5, is based on data from the
    Second HSC Data Release, details of which can be found here:
    Aihara, H., AlSayyad, Y., Ando, M., et al. 2019, PASJ, 71, 114
    doi: 10.1093/pasj/psz103
    """
    def_ratio_file = os.path.join(RAILDIR, "rail/examples/creation/data/hsc_ratios_and_specz.hdf5")
    def_set_file = os.path.join(RAILDIR, "rail/examples/creation/data/HSC_grid_settings.pkl" )

    name = 'GridSelection'
    config_options = Degrader.config_options.copy()
    config_options.update(color_redshift_cut=Param(bool, True, msg='using color-based redshift cut'),
                          percentile_cut=Param(float, 99.0, msg='percentile cut-off for each pixel in color-based redshift cut off'),
                          redshift_cut=Param(float, 100.0, msg="cut redshifts above this value"),
                          ratio_file=Param(str, def_ratio_file,
                                           msg="path to ratio file"),
                          settings_file=Param(str, def_set_file,
                                              msg='path to pickled parameters file'),
                          random_seed=Param(int, 12345, msg="random seed for reproducibility"),
                          scaling_factor=Param(float, 1.588, msg="multiplicative factor for ratios to adjust number of galaxies kept"))

    def __init__(self, args, comm=None):

        Degrader.__init__(self, args, comm=comm)

        if self.config.redshift_cut < 0:
            raise ValueError("redshift cut must be positive")
        if (self.config.percentile_cut < 0) | (self.config.percentile_cut > 100):
            raise ValueError('percentile cut off must be between 0 and 100')

    def run(self):
        """
        HSC galaxies were binned in color magnitude space with i-band mag from -2 to 6 and g-z color from 13 to 26
        200 bins in each direction. The ratio of of galaxies with spectroscopic redshifts (training galaxies) to
        galaxies with only photometry in HSC wide field (application galaxies) was computed for each pixel. We divide
        the data into the same pixels and randomly select galaxies into the training sample based on the HSC ratios.
        If using a color-based redshift cut, galaxies with redshifts > the percentile cut are removed from the sample
        before making the random selection.
        """
        np.random.seed(self.config.random_seed)

        data = self.get_data('input')
        with open(self.config.settings_file, 'rb') as handle:
            settings = pickle.load(handle)
        # check settings values
        check_vals = ['x_band_1', 'x_band_2', 'y_band_1', 'y_band_2', 'x_limits', 'y_limits']
        for val in check_vals:
            if val not in settings.keys():  # pragma: no cover
                raise KeyError(f"required key {val} not present in {self.config.settings_file}!")
        for val in check_vals[:-2]:
            if settings[val] != '' and settings[val] not in data.keys():  # pragma: no cover
                raise KeyError(f"column {settings[val]} not present in data file!")

        if settings['x_band_2'] == '':
            data['x_vals'] = data[settings['x_band_1']].to_numpy()
        else:  # pragma: no cover
            data['x_vals'] = (data[settings['x_band_1']] - data[settings['x_band_2]']]).to_numpy()

        if settings['y_band_2'] == '':  # pragma: no cover
            data['y_vals'] = data[settings['y_band_1']].to_numpy()
        else:
            data['y_vals'] = (data[settings['y_band_1']] - data[settings['y_band_2']]).to_numpy()

        # now remove galaxies that don't fall in i-mag and g-z color range of HSC data. These will always be application sample galaxies
        x_lims = settings['x_limits']
        y_lims = settings['y_limits']
        if len(x_lims) != 2:  # pragma: no cover
            raise ValueError("x limits should have two elements!")
        if len(y_lims) != 2:  # pragma: no cover
            raise ValueError("y limits should have two elements!")
        data_hsc_like = data[(data['x_vals'] >= x_lims[0]) & (data['x_vals'] <= x_lims[1]) & (data['y_vals'] >= y_lims[0]) & (data['y_vals'] <= y_lims[1])]

        x_vals = data_hsc_like['x_vals'].to_numpy()
        y_vals = data_hsc_like['y_vals'].to_numpy()

        # Opens file with HSC ratios
        ratios = tables_io.read(self.config.ratio_file)['ratios']

        # Sets pixel edges to be the same as were used to calculate HSC ratios
        num_edges_y = ratios.shape[1] + 1
        num_edges_x = ratios.shape[0] + 1
        y_edges = np.linspace(y_lims[0], y_lims[1], num_edges_y)
        x_edges = np.linspace(x_lims[0], x_lims[1], num_edges_x)

        # Calculate the max spec-z for each pixel from percentile_cut
        # If not using color-based redshift cut, max spec-z set to 100
        # percentiles are determined from the HSC data, not the input
        # data, which gives smoother distributions in many cases.
        if not self.config.color_redshift_cut:  # pragma: no cover
            max_specz = np.ones_like(ratios) * 100.
        else:
            percentile_cut = self.config['percentile_cut']
            hsc_spec_galaxies = pd.read_hdf(self.config['ratio_file'], 'data', 'r')

            hsc_spec_mags = hsc_spec_galaxies['mag'].to_numpy()
            hsc_spec_colors = hsc_spec_galaxies['color'].to_numpy()

            pixels_hsc_colors = np.searchsorted(y_edges, hsc_spec_colors) - 1
            pixels_hsc_mags = np.searchsorted(x_edges, hsc_spec_mags) - 1

            pixel_tot = (len(x_edges) - 1) * pixels_hsc_mags + pixels_hsc_colors
            hsc_spec_galaxies['total_pixel'] = pixel_tot  # tags each galaxy with a single pixel number instead of one color and one magnitude

            unique_pixels = hsc_spec_galaxies['total_pixel'].unique()

            max_specz = np.zeros((len(x_edges) - 1, len(y_edges) - 1))
            for i in range(len(x_edges) - 1):
                for j in range(len(y_edges) - 1):
                    pixel = i * int(len(x_edges) - 1) + j
                    if np.isin(pixel, unique_pixels):
                        temp_dict = hsc_spec_galaxies[hsc_spec_galaxies['total_pixel'] == pixel]
                        spec_zs = temp_dict['specz'].to_numpy()
                        percentile = np.percentile(spec_zs, percentile_cut)
                        max_specz[i][j] = percentile

        # For each galaxy in data, identifies the pixel it belongs in, and adds the ratio for that pixel to a new column in data called 'ratios'
        pixels_y = np.searchsorted(y_edges, y_vals) - 1
        pixels_x = np.searchsorted(x_edges, x_vals) - 1

        ratio_list = []
        max_specz_list = []
        for pix_x, pix_y in zip(pixels_x, pixels_y):
            ratio_list.append(ratios[pix_y][pix_x])
            max_specz_list.append(max_specz[pix_y][pix_x])

        data_hsc_like['ratios'] = ratio_list
        data_hsc_like['max_specz'] = max_specz_list

        # remove galaxies with redshifts higher than the color-based cutoff
        data_hsc_like_redshift_cut = data_hsc_like[data_hsc_like['redshift'] <= data_hsc_like['max_specz']]

        # If making a redshift cut, do that now
        if self.config['redshift_cut'] != 100:  # pragma: no cover
            data_hsc_like_redshift_cut = data_hsc_like_redshift_cut[data_hsc_like_redshift_cut['redshift'] <= self.config['redshift_cut']]

        # This picks galaxies for the training set
        unique_ratios = data_hsc_like['ratios'].unique()

        keep_inds = []

        if self.config.color_redshift_cut:
            # multiplicative factor that can account for the fact that we select fewer galaxies
            # with the color-based redshift cut after percentile cut is applied
            factor = self.config.scaling_factor
        else:  # pragma: no cover
            factor = 1
        for xratio in unique_ratios:
            temp_data = data_hsc_like_redshift_cut[data_hsc_like_redshift_cut['ratios'] == xratio]
            number_to_keep = len(temp_data) * xratio
            if number_to_keep * factor <= len(temp_data):
                number_to_keep = number_to_keep * factor
            else:  # pragma: no cover
                number_to_keep = len(temp_data)

            if int(number_to_keep) != number_to_keep:
                random_num = np.random.uniform()
            else:
                random_num = 2

            number_to_keep = np.floor(number_to_keep)
            indices_to_list = list(temp_data.index.values)
            np.random.shuffle(indices_to_list)

            if random_num > xratio:  # pragma: no cover
                for j in range(int(number_to_keep)):
                    keep_inds.append(indices_to_list[j])

            else:  # pragma: no cover
                for j in range(int(number_to_keep) + 1):
                    keep_inds.append(indices_to_list[j])

        training_data = data_hsc_like_redshift_cut.loc[keep_inds, :]
        training_data = training_data[training_data['redshift'] > 0]
        training_data = training_data.drop(['x_vals', 'y_vals', 'ratios'], axis=1)

        self.add_data('output', training_data)
