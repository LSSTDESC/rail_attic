"""Degraders that emulate spectroscopic effects on photometry"""

import numpy as np
import pandas as pd
import pickle
import tables_io
from rail.creation.degradation import Degrader
from ceci.config import StageParameter as Param


class LineConfusion(Degrader):
    """Degrader that simulates emission line confusion.

    Example: degrader = LineConfusion(true_wavelen=3727,
                                      wrong_wavelen=5007,
                                      frac_wrong=0.05)
    is a degrader that misidentifies 5% of OII lines (at 3727 angstroms)
    as OIII lines (at 5007 angstroms), which results in a larger
    spectroscopic redshift .

    Note that when selecting the galaxies for which the lines are confused,
    the degrader ignores galaxies for which this line confusion would result
    in a negative redshift, which can occur for low redshift galaxies when
    wrong_wavelen < true_wavelen.

    Parameters
    ----------
    true_wavelen : positive float
        The wavelength of the true emission line.
        Wavelength unit assumed to be the same as wrong_wavelen.
    wrong_wavelen : positive float
        The wavelength of the wrong emission line, which is being confused
        for the correct emission line.
        Wavelength unit assumed to be the same as true_wavelen.
    frac_wrong : float between zero and one
        The fraction of galaxies with confused emission lines.
    """

    name = 'LineConfusion'
    config_options = Degrader.config_options.copy()
    config_options.update(true_wavelen=float,
                          wrong_wavelen=float,
                          frac_wrong=float)

    def __init__(self, args, comm=None):
        """
        """
        Degrader.__init__(self, args, comm=comm)
        # validate parameters
        if self.config.true_wavelen < 0:
            raise ValueError("true_wavelen must be positive, not {self.config.true_wavelen}")
        if self.config.wrong_wavelen < 0:
            raise ValueError("wrong_wavelen must be positive, not {self.config.wrong_wavelen}")
        if self.config.frac_wrong < 0 or self.config.frac_wrong > 1:
            raise ValueError("frac_wrong must be between 0 and 1., not {self.config.wrong_wavelen}")

    def run(self):
        """ Run method

        Applies line confusion

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Puts the data into the data store under this stages 'output' tag
        """
        data = self.get_data('input')

        # convert to an array for easy manipulation
        values, columns = data.values.copy(), data.columns.copy()

        # get the minimum redshift
        # if wrong_wavelen < true_wavelen, this is minimum the redshift for
        # which the confused redshift is still positive
        zmin = self.config.wrong_wavelen / self.config.true_wavelen - 1

        # select the random fraction of galaxies whose lines are confused
        rng = np.random.default_rng(self.config.seed)
        idx = rng.choice(
            np.where(values[:, 0] > zmin)[0],
            size=int(self.config.frac_wrong * values.shape[0]),
            replace=False,
        )

        # transform these redshifts
        values[idx, 0] = (
            1 + values[idx, 0]
        ) * self.config.true_wavelen / self.config.wrong_wavelen - 1

        # return results in a data frame
        outData = pd.DataFrame(values, columns=columns)
        self.add_data('output', outData)


class InvRedshiftIncompleteness(Degrader):
    """Degrader that simulates incompleteness with a selection function
    inversely proportional to redshift.

    The survival probability of this selection function is
    p(z) = min(1, z_p/z),
    where z_p is the pivot redshift.

    Parameters
    ----------
    pivot_redshift : positive float
        The redshift at which the incompleteness begins.
    """

    name = 'InvRedshiftIncompleteness'
    config_options = Degrader.config_options.copy()
    config_options.update(pivot_redshift=float)

    def __init__(self, args, comm=None):
        """
        """
        Degrader.__init__(self, args, comm=comm)
        if self.config.pivot_redshift < 0:
            raise ValueError("pivot redshift must be positive, not {self.config.pivot_redshift}")

    def run(self):
        """ Run method

        Applies incompleteness

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Puts the data into the data store under this stages 'output' tag
        """
        data = self.get_data('input')

        # calculate survival probability for each galaxy
        survival_prob = np.clip(self.config.pivot_redshift / data["redshift"], 0, 1)

        # probabalistically drop galaxies from the data set
        rng = np.random.default_rng(self.config.seed)
        mask = rng.random(size=data.shape[0]) <= survival_prob

        self.add_data('output', data[mask])


class HSCSelection(Degrader):
    """
    Uses the ratio of HSC spectroscpic galaxies to photometric galaxies to portion a sample
    into training and application samples. Option of further degrading the training sample by limiting it to galaxies
    less than a redshift cutoff by specifying redshift_cut.

    configuration options:
    redshift_cut: redshift above which all galaxies will be removed from training sample. Default is 100
    ratio_file: hdf5 file containing an array of spectroscpic vs. photometric galaxies in each pixel. Default is hsc_ratios.hdf5 for an HSC based selection
    settings_file: pickled dictionary containing information about colors and magnitudes used in defining the pixels. Dictionary must include the following keys:
         'mag_band': string, this is the band used for the magnitude in the color magnitude diagram. Default for HSC is 'i',
         'color_band_blue': string, this is the bluer band used for the color in the color magnitude diagram. Default for HSC is 'g',
         'color_band_red': string, this is the redder band used for the color in the color magnitude diagram. Default for HSC is 'z',
         'mag_lims': list, this is a list of the lower and upper limits of the magnitude. Default for HSC is [13, 16],
         'color_lims': list, this is a list of the lower and upper limits of the color. Default for HSC is [-2, 6]}
    """

    name = 'HSCSelection'
    config_options = Degrader.config_options.copy()
    config_options.update(redshift_cut=Param(float, 100.0, msg="cut redshifts above this value"),
                          ratio_file=Param(str, './examples/creation/data/hsc_ratios.hdf5',
                                           msg="path to ratio file"),
                          settings_file=Param(str, './examples/creation/data/hsc_config_settings.pkl',
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

        mag_band = settings['mag_band']
        blue_band = settings['color_band_blue']
        red_band = settings['color_band_red']

        # first, add colors to data
        g_mag = data[blue_band].to_numpy()
        z_mag = data[red_band].to_numpy()

        gz_color = g_mag - z_mag

        data['color'] = gz_color

        # now remove galaxies that don't fall in i-mag and g-z color range of HSC data. These will always be application sample galaxies
        mag_lims = settings['mag_lims']
        color_lims = settings['color_lims']
        data_hsc_like = data[(data[mag_band] >= mag_lims[0]) & (data[mag_band] <= mag_lims[1]) & (data['color'] >= color_lims[0]) & (data['color'] <= color_lims[1])]

        i_mag = data_hsc_like[mag_band].to_numpy()
        color = data_hsc_like['color'].to_numpy()

        # Opens file with HSC ratios
        ratios = tables_io.read(self.config.ratio_file)['ratios']

        # Sets pixel edges to be the same as were used to calculate HSC ratios
        num_edges_colors = ratios.shape[1]
        num_edges_mags = ratios.shape[0]
        colors_edges = np.linspace(color_lims[0], color_lims[1], num_edges_colors)
        mags_edges = np.linspace(mag_lims[0], mag_lims[1], num_edges_mags)

        # For each galaxy in data, identifies the pixel it belongs in, and adds the ratio for that pixel to a new column in data called 'ratios'
        pixels_colors = np.searchsorted(colors_edges, color)
        pixels_mags = np.searchsorted(mags_edges, i_mag)

        pixels_colors = pixels_colors - 1
        pixels_mags = pixels_mags - 1

        ratio_list = []
        for i in range(len(pixels_colors)):
            ratio_list.append(ratios[pixels_colors[i]][pixels_mags[i]])

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

        # For the pessimistic choice, also remove galaxies with z > redshift_cut from the sample
        if self.config['redshift_cut'] != 100:
            training_data = training_data[training_data['redshift'] <= self.config['redshift_cut']]

        training_data = training_data.drop(['color', 'ratios'], axis=1)

        self.add_data('output', training_data)
