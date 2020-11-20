
import numpy as np
import pandas as pd

class Creator():
    """
    An object that supplies mock data for redshift estimation experiments.
    The mock data is drawn from a probability distribution defined by the generator,
    with an optional selection function applied.

    generator - an object defining a redshift probability distribution. Must have
                a sample and log_prob methods.
    selection_fn - a selection function to apply to the generated sample
    params - additional information desired to be stored with the instance as a dictionary
    """

    def __init__(self, generator, selection_fn=None, params=None):
        self.generator = generator
        self.selection_fn = selection_fn
        self.params = params

    def sample(self, n_samples, seed=None, include_err=False, include_pdf=False, zmin=0, zmax=2, dz=0.02):
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
            # and 10 yr limiting mags from https://www.lsst.org/scientists/keynumbers
            # currently these limitting mags are set 2 mags dimmer than actual LSST vals
            err_params = {'gamma_u':0.038,
                          'gamma_g':0.039,
                          'gamma_r':0.039,
                          'gamma_i':0.039,
                          'gamma_z':0.039,
                          'gamma_y':0.039,
                          'm5_u':28.1,
                          'm5_g':29.4,
                          'm5_r':29.5,
                          'm5_i':28.8,
                          'm5_z':28.1,
                          'm5_y':26.9}

            for band in ['u','g','r','i','z','y']:
                gamma = err_params[f'gamma_{band}']
                m5 = err_params[f'm5_{band}']
                x = 10**(0.4*(sample[band]-m5))
                sample[f'{band}err'] = np.sqrt( (0.04 - gamma) * x + gamma * x**2 )
                sample[band] = np.random.normal(sample[band], sample[f'{band}err'])
        
        # calculate conditional pdfs
        if include_pdf: 
            posteriors = self.generator.pz_estimate(sample, zmin=zmin, zmax=zmax, dz=dz, convolve_err=include_err)
            sample['pz_pdf'] = list(posteriors)

        return sample

        