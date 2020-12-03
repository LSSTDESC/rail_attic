import numpy as np
import pandas as pd

import rail.creation.utils as rcu

class Creator():
    """
    An object that supplies mock data for redshift estimation experiments.
    The mock data is drawn from a probability distribution defined by the generator, with an optional selection function applied.
    """

    def __init__(self, generator, selection_fn=None, params=None):
        """
        Parameters
        ----------
        generator: rail.Generator object
            object defining a redshift probability distribution.
            Must have sample, log_prob and pz_estimate methods (see generator.py)
        selection_fn: function, optional
            a selection function to apply to the generated sample
            Must take a pandas dataframe and a seed number, and return a pandas dataframe with the selection function applied.
        params: dictionary, optional
            additional information desired to be stored with the instance as a dictionary.
            By default, includes error params for Eq 5 from https://arxiv.org/pdf/0805.2366.pdf
        """
        self.generator = generator
        self.selection_fn = selection_fn

        # if no params provided, use defaults
        if params is None:
            self.params = {}
            self.params['bands'] = rcu.param_defaults['bands']
            self.params['err_params'] = rcu.param_defaults['err_params']
        # if params are provided, fill in defaults if bands and err_params not included
        else:
            self.params = params
            param_keys = self.params.keys()
            for key_name in ['bands', 'err_params']:
                if key_name not in param_keys:
                    self.params[key_name] = rcu.param_defaults[key_name]

        # Basic sanity check on params values
        err_str = "Number of err_params is not equal to 2x the number of bands"
        num_bands = len(self.params['bands'])
        num_err_params = len(self.params['err_params'])
        assert (2*num_bands == num_err_params), err_str

    def sample(self, n_samples, seed=None, include_err=False, err_seed=None,
               include_pdf=False, zinfo=None):
        """
        Draws n_samples from the generator

        Parameters
        ----------
        n_samples: int
            number of samples to draw
        include_err: boolean, optional
            if True, errors are calculated using Eq 5 from https://arxiv.org/pdf/0805.2366.pdf
        include_pdf: boolean, optional
            if True, then posteriors are returned for each galaxy.
            The posteriors are saved in the column pz_pdf, and the redshift grid saved as df.attrs['pz_grid'].
            The grid is defined by np.arange(zmin, zmax+dz, dz)
        seed: int, optional
            sets the random seed for drawing samples
        err_seed: int, optional
            sets the seed for the Gaussian errors
        zinfo: dictionary, optional
            must contain `zmin`, `zmax`, `dz` floats

        Returns
        -------
        sample: pandas DataFrame
            samples from model, containing photometry, true redshift, and posterior PDF if requested

        Notes
        -----
        We should rename `sample` output to not be the same as the name of the method `sample()`.
        Output posterior format is currently hardcoded to grid evaluations but could be integrated with qp.
        We will probably change the output format to dovetail with the evaluation module when ready.
        """

        rng = np.random.default_rng(seed)

        # get samples
        sample = self.generator.sample(n_samples, seed=rng.integers(1e18))

        if self.selection_fn:
            # apply selection function
            sample = self.selection_fn(sample, seed=rng.integers(1e18))
            # calculate fraction that survives the cut
            selected_frac = len(sample)/n_samples
            # draw more samples and cut until we have enough samples
            while len(sample) < n_samples:
                # estimate how many extras to draw
                n_supplement = int( 1.1/selected_frac * (n_samples - len(sample)) )
                # draw new samples and apply cut
                new_sample = self.generator.sample(n_supplement, seed=rng.integers(1e18))
                new_sample = self.selection_fn(new_sample, seed=rng.integers(1e18))
                # add these to the larger set
                sample = pd.concat((sample, new_sample), ignore_index=True)
            # cut out the extras
            sample = sample.iloc[:n_samples]

        if include_err:
            # add errors to the sample
            # using Eq 5 from https://arxiv.org/pdf/0805.2366.pdf
            rng = np.random.default_rng(err_seed)

            # calculate error in each band, then add Gaussian errors
            for band in self.params['bands']:
                gamma = self.params['err_params'][f'gamma_{band}']
                m5 = self.params['err_params'][f'm5_{band}']
                x = 10**(0.4 * (sample[band] - m5))
                sample[f'{band}err'] = np.sqrt((0.04 - gamma) * x + gamma * x**2)
                sample[band] = rng.normal(sample[band], sample[f'{band}err'])
        if not zinfo:
            zinfo = rcu.zinfo

        # calculate posteriors
        if include_pdf:
            nfeatures = len(self.params['bands']) + 1
            posteriors = self.generator.pz_estimate(sample.iloc[:,:nfeatures], zinfo=zinfo)
            sample.attrs['pz_grid'] = np.arange(zinfo['zmin'], zinfo['zmax'] + zinfo['dz'], zinfo['dz'])
            sample['pz_pdf'] = list(posteriors)

        return sample
