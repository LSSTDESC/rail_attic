"""Degraders that emulate spectroscopic effects on photometry"""

import numpy as np
import pandas as pd
from rail.creation.degradation import Degrader


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
    """

    name='HSCSelection'
    config_options = Degrader.config_options.copy()
    config_options.update(**{'redshift_cut':100})

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

        data = self.get_data('input')
        
        #first, add colors to data
        g_mag = data['g'].to_numpy()
        z_mag = data['z'].to_numpy()

        gz_color = g_mag-z_mag
        
        data['color'] = gz_color
        
        #now remove galaxies that don't fall in i-mag and g-z color range of HSC data. These will always be application sample galaxies
        
        data_hsc_like = data[(data['i'] >= 13) & (data['i'] <=26) & (data['color'] >= -2) & (data['color'] <=6)]
        
        i_mag = data_hsc_like['i'].to_numpy()
        color = data_hsc_like['color'].to_numpy()

       
        #Sets pixel edges to be the same as were used to calculate HSC ratios
        x_edges = np.linspace(-2, 6, 201)
        y_edges = np.linspace(13, 26, 201)

        #Opens file with HSC ratios
        ratios = pd.read_csv('hsc_ratios.csv', delimiter=',', header=None)
        
        #For each galaxy in data, identifies the pixel it belongs in, and adds the ratio for that pixel to a new column in data called 'ratios'
        pixels_x = np.searchsorted(x_edges, color)
        pixels_y = np.searchsorted(y_edges, i_mag)
        
        pixels_x = pixels_x-1
        pixels_y = pixels_y-1
        
        ratio_list  = []
        for i in range(len(pixels_x)):
            ratio_list.append(ratios[pixels_y[i]][pixels_x[i]])
        

        data_hsc_like['ratios'] = ratio_list

        #This picks galaxies for the training set
        unique_ratios = data_hsc_like['ratios'].unique()

        keep_inds = []
        for i in range(len(unique_ratios)):
            temp_data = data_hsc_like[data_hsc_like['ratios'] == unique_ratios[i]]
            number_to_keep = int(len(temp_data)*unique_ratios[i])
            indices_to_list = temp_data.index.values
            for j in range(0, number_to_keep):
                keep_inds.append(indices_to_list[j])
 

        training_data = data_hsc_like.loc[keep_inds,:]

        #For the pessimistic choice, also remove galaxies with z > redshift_cut from the sample
        if self.config['redshift_cut'] != 100:
            training_data = training_data[training_data['redshift'] <= self.config['redshift_cut']]
        
        training_data = training_data.drop(['color', 'ratios'], axis=1)

        self.add_data('output', training_data)
