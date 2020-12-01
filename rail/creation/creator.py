import numpy as np


class Creator():
    """
    An object that supplies mock data for redshift estimation experiments.
    The mock data is drawn from a probability distribution defined by the generator,
    with an optional selection function applied.

    generator - an object defining a redshift probability distribution. Must have
                a sample, log_prob and pz_estimate methods (see generator.py)
    selection_fn - a selection function to apply to the generated sample
    params - additional information desired to be stored with the instance as a dictionary
    """

    def __init__(self, generator, selection_fn=None, params=None):
        self.generator = generator
        self.selection_fn = selection_fn

        param_defaults = {}
        param_defaults['bands'] = ['u', 'g', 'r', 'i', 'z', 'y']

        # 10 yr limiting mags from https://www.lsst.org/scientists/keynumbers
        # currently these limitting mags are set 2 mags
        # dimmer than actual LSST vals
        param_defaults['err_params'] = {'gamma_u': 0.038,
                                        'gamma_g': 0.039,
                                        'gamma_r': 0.039,
                                        'gamma_i': 0.039,
                                        'gamma_z': 0.039,
                                        'gamma_y': 0.039,
                                        'm5_u': 28.1,
                                        'm5_g': 29.4,
                                        'm5_r': 29.5,
                                        'm5_i': 28.8,
                                        'm5_z': 28.1,
                                        'm5_y': 26.9}

        if params is None:
            self.params = {}
            self.params['bands'] = param_defaults['bands']
            self.params['err_params'] = param_defaults['err_params']
        else:
            self.params = params
            param_keys = self.params.keys()
            for key_name in ['bands', 'err_params']:
                if key_name not in param_keys:
                    self.params[key_name] = param_defaults[key_name]

    def sample(self, n_samples, seed=None, include_err=False,
               include_pdf=False, zmin=0, zmax=2, dz=0.02):
        """
        Draw n_samples from the generator.
        If include_pdf == True, then a conditional pdf for redshift will be returned with each
        set of galaxy magnitudes. The pdf is evaluated at points in np.arange(zmin, zmax+dz, dz).
        seed sets the random seed for drawing samples.
        """

        # get samples
        sample = self.generator.sample(n_samples, seed=seed)

        if include_err:
            # add errors to the sample
            # using Eq 5 from https://arxiv.org/pdf/0805.2366.pdf

            for band in self.params['bands']:
                gamma = self.params['err_params'][f'gamma_{band}']
                m5 = self.params['err_params'][f'm5_{band}']
                x = 10**(0.4*(sample[band]-m5))
                sample[f'{band}err'] = np.sqrt((0.04 - gamma) * x + gamma * x**2)
                sample[band] = np.random.normal(sample[band],
                                                sample[f'{band}err'])
       
        # calculate conditional pdfs
        if include_pdf:
            nfeatures = len(self.params['bands']) + 1
            posteriors = self.generator.pz_estimate(sample.iloc[:,:nfeatures], zmin=zmin, zmax=zmax, dz=dz)
            sample.attrs['pz_grid'] = np.arange(zmin, zmax+dz, dz)
            sample['pz_pdf'] = list(posteriors)

        return sample
