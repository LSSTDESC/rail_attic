"""
A superclass for creators of mock data for redshift estimation experiments
"""

from __future__ import absolute_import
__all__ = ['Creator']


class Creator(object):

    def __init__(self, file, generatorClass, sel_fcn = None, params=None):
        """
        An object that supplies mock data for redshift estimation experiments from a smooth space
        Parameters
        ----------
        file   : smoothly interpolated space written to pickle file to be sampled for photometry
        TF     : transfer function
        sel_fcn: selection function as written to a pickle file
        params : additional parameters desired to be stored with the instance as a dictionary

        Notes
        -----
        Currently the Creator class does not generate the smoothly interpolated space from the DC2 data, but reads it in.
        """

        self.file = file
        self.generator = generatorClass
        self.generator.load_model(file)

    def sample_pspace(self, n_gals=1e6, output_dims=('zTrue', 'g', 'r', 'i', 'z', 'y', 'gErr', 'rErr', 'iErr', 'zErr', 'yErr')):
        """

        Parameters
        ----------
        n_gals      : the number of galaxies in the desired sample
        output_dims : the desired fields for the sample

        Output
        ------
        A RAILinfo object with the prescribed fields drawn from the smooth space described by the file read in at initiation of the instance
        and stored in self.dataspace.

        Notes
        -----
        formulism may change after RAILinfo gets more developed
        """
        #get the sample from file
        print('Drawing {0} samples from {1}.'.format(n_gals,self.file))

        #store sample in rail.RAILinfo object, identifying the data frame as photometry and the metadata as other relevant parameters
        #sample = rail.RAILinfo(...)
        #sample.metadata =
        #sample.data =
        sample = self.generator.return_sample(n_gals)

        return sample
