
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

    def sample(self, n_samples, include_pdf=False, zmin=0, zmax=2.5, dz=0.02, seed=None):
        """
        Draw n_samples from the generator. 
        If include_pdf == True, then a conditional pdf for redshift will be returned with each
        set of galaxy magnitudes. The pdf is evaluated at points in np.arange(zmin, zmax+dz, dz).
        seed sets the random seed for drawing samples.
        """
        
        # get samples
        sample = self.generator.sample(n_samples, seed=seed)
        
        # apply selection function
        sample = self.selection_fn(sample) if self.selection_fn else sample
        
        # calculate conditional pdfs
        if include_pdf:
            
            # generate the redshift grid
            zs = np.arange(zmin, zmax+dz, dz)
            
            # log(pdf) for each galaxy in the sample
            log_pz = self.generator.log_prob(pd.DataFrame({'redshift': np.tile(zs, len(sample)),
                                                           'u': np.repeat(sample['u'], len(zs)),
                                                           'g': np.repeat(sample['g'], len(zs)),
                                                           'r': np.repeat(sample['r'], len(zs)),
                                                           'i': np.repeat(sample['i'], len(zs)),
                                                           'z': np.repeat(sample['z'], len(zs)),
                                                           'y': np.repeat(sample['y'], len(zs))}))
            
            # reshape so each row is a galaxy
            log_pz = log_pz.reshape((len(sample), -1))
            # calculate the pdf
            pz = np.exp(log_pz) / (np.exp(log_pz) * dz).sum(axis=1).reshape(-1,1)
            
            # save the redshift grid in the dataframe metadata
            sample.attrs['pdf_z'] = zs
            # save all the pdfs
            sample['pdf'] = list(pz)
                                   
        return sample

        