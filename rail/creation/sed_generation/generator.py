""" Abstract base class defining a generator

The key feature is that the __call__ method takes a TableHandle and
and returns a FitsHandle, and wraps the run method
"""

from rail.core.stage import RailStage
from rail.core.data import TableHandle, FitsHandle


class SedGenerator(RailStage):
    """Base class SedGenerator, which generates synthetic rest-frame SEDs

    Generators take "input" data in the form of files passed to table_io and
    provide as "output" another table_io readable file
    """

    name = 'Generator'
    config_options = RailStage.config_options.copy()
    config_options.update(seed=12345)
    inputs = [('input', TableHandle)]
    outputs = [('output', FitsHandle)]

    def __init__(self, args, comm=None):
        """Initialize Generator that can create rest-frame SEDs"""
        RailStage.__init__(self, args, comm=comm)

    def __call__(self, sample, seed: int = None, physical_units=True, tabulated_sfh_file=None, tabulated_lsf_file=None):
        """The main interface method for `Generator`

        Generate SEDs.

        This will attach the sample to this `Generator`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Estimator
        by using `self.add_data('output', output_data)`.

        Finally, this will return a TableHandle providing access to that output data.

        Parameters
        ----------
        sample : astropy.table
            galaxy physical properties
        seed : int, default=None
            An integer to set the numpy random seed

        Returns
        -------
        output_data : `TableHandle`
            A handle giving access to an astropy.table with rest-frame SEDs
        """

        self.config.physical_units = physical_units

        if seed is not None:
            self.config.seed = seed
        if tabulated_sfh_file is not None:
            self.config.tabulated_sfh_file = tabulated_sfh_file
        else:
            self.config.tabulated_sfh_file = None
        if tabulated_lsf_file is not None:
            self.config.tabulated_lsf_file = tabulated_lsf_file
        else:
            self.config.tabulated_lsf_file = None

        self.set_data('input', sample)
        self.run()
        self.finalize()

        return self.get_handle('output')
