"""Degraders that emulate spectroscopic effects on photometry"""

import numpy as np
import pickle
import tables_io
from rail.creation.degradation import Degrader
from ceci.config import StageParameter as Param


class GridSelection(Degrader):
    """
    Uses the ratio of HSC spectroscpic galaxies to photometric galaxies to portion a sample
    into training and application samples. Option of further degrading the training sample by limiting it to galaxies
    less than a redshift cutoff by specifying redshift_cut.

    configuration options:
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
    """

    name = 'GridSelection'
    config_options = Degrader.config_options.copy()
    config_options.update(redshift_cut=Param(float, 100.0, msg="cut redshifts above this value"),
                          ratio_file=Param(str, './examples/creation/data/hsc_ratios.hdf5',
                                           msg="path to ratio file"),
                          settings_file=Param(str, './examples/creation/data/HSC_grid_settings.pkl',
                                              msg='path to pickled parameters file'),
                          random_seed=Param(int, 12345, msg="random seed for reproducibility"))

    def __init__(self, args, comm=None):

        Degrader.__init__(self, args, comm=comm)

        if self.config.redshift_cut < 0:
            raise ValueError("redshift cut must be positive")

    def run(self):
        """
        HSC galaxies were binned in color magnitude space with i-band mag from -2 to 6 and g-z color from 13 to 26
        200 bins in each direction. The ratio of of galaxies with spectroscopic redshifts (training galaxies) to
        galaxies with only photometry in HSC wide field (application galaxies) was computed for each pixel. We divide
        the data into the same pixels and randomly select galaxies into the training sample based on the HSC ratios
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

        # For each galaxy in data, identifies the pixel it belongs in, and adds the ratio for that pixel to a new column in data called 'ratios'
        pixels_y = np.searchsorted(y_edges, y_vals)
        pixels_x = np.searchsorted(x_edges, x_vals)

        pixels_y = pixels_y - 1
        pixels_x = pixels_x - 1

        ratio_list = []
        for i in range(len(pixels_y)):
            ratio_list.append(ratios[pixels_y[i]][pixels_x[i]])

        data_hsc_like['ratios'] = ratio_list

        # This picks galaxies for the training set
        unique_ratios = data_hsc_like['ratios'].unique()

        keep_inds = []
        for xratio in unique_ratios:
            temp_data = data_hsc_like[data_hsc_like['ratios'] == xratio]
            numingroup = len(temp_data)
            randoms = np.random.uniform(size=numingroup)
            mask = (randoms <= xratio)
            keepers = temp_data[mask].index.values
            keep_inds.append(keepers)

        # our indeces are a list of list, need to flatten that out so that we can easily apply to the dataframe
        flat_inds = [item for sublist in keep_inds for item in sublist]
        training_data = data_hsc_like.loc[flat_inds, :]

        training_data = training_data[training_data['redshift'] > 0]
        # For the pessimistic choice, also remove galaxies with z > redshift_cut from the sample
        if not np.isclose(self.config['redshift_cut'], 100.):
            training_data = training_data[training_data['redshift'] <= self.config['redshift_cut']]

        training_data = training_data.drop(['x_vals', 'y_vals', 'ratios'], axis=1)

        self.add_data('output', training_data)
