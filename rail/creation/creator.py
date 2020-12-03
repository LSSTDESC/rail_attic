import numpy as np
import pandas as pd


class Creator():
    """
    An object that supplies mock data for redshift estimation experiments.
    The mock data is drawn from a probability distribution defined by the generator,
    with an optional selection function applied.

    generator - an object defining a redshift probability distribution. Must have
                a sample, log_prob and pz_estimate methods (see generator.py)
    selection_fn - a selection function to apply to the generated sample. Must take a pandas
                dataframe and a seed number, and return a pandas dataframe with the selection
                function applied.
    params - additional information desired to be stored with the instance as a dictionary.
    """

    def __init__(self, generator, selection_fn=None, params=None):
        self.generator = generator
        self.selection_fn = selection_fn
        self.params = params

    def sample(self, n_samples, seed=None,
               include_pdf=False, zmin=0, zmax=2, dz=0.02):
        """
        Draw n_samples from the generator.
        include_pdf - if True, then posteriors are returned for each galaxy. The posteriors are saved
                        in the column pz_pdf, and the redshift grid saved as df.attrs['pz_grid'].
                        The grid is defined by np.arange(zmin, zmax+dz, dz)
        seed - sets the random seed for drawing samples
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

        # calculate posteriors
        if include_pdf:
            posteriors = self.generator.pz_estimate(sample, zmin=zmin, zmax=zmax, dz=dz)
            sample.attrs['pz_grid'] = np.arange(zmin, zmax+dz, dz)
            sample['pz_pdf'] = list(posteriors)

        return sample
