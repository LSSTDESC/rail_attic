""" Abstract base class and derived class defining a generator for galaxy population galaxy_modelling components

The key feature is that the __call__ method takes a configuration file and
returns a FitsHandle, and wraps the run method
"""

from rail.core.stage import RailStage
from rail.core.data import FitsHandle
from ceci.config import StageParameter as Param
import gal_pop_model_components
from gal_pop_model_components import galaxy_population_modelling


class GalaxyGenerator(RailStage):
    """
    Base class for generating galaxy populations from library of galaxy_modelling components.

    It takes as "input" a configuration file and provides as "output" a fits table compatible with the input
    accepted by rail_fsps.
    """

    name = 'Galaxy Generator'
    config_options = RailStage.config_options.copy()
    config_options.update(seed=12345)
    inputs = [('input', '')]
    outputs = [('output', FitsHandle)]

    def __init__(self, args, comm=None):
        """
        Initialize GalaxyGenerator
        """

        RailStage.__init__(self, args, comm=comm)

    def __call__(self, sample, seed: int = None):  # pragma: no cover
        """The main interface method for `GalaxyGenerator`

        Generate galaxy population from library of galaxy_modelling components.

        set_data will attach the sample to this `GalaxyGenerator`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Estimator
        by using `self.add_data('output', output_data)`.

        Finally, this will return a TableHandle providing access to that output data.

        Parameters
        ----------
        sample : configuration file
            which galaxy_modelling components to use
        seed : int, default=None
            An integer to set the numpy random seed

        Returns
        -------
        output_data : `TableHandle`
            A handle giving access to an astropy.table with galaxy population properties compatible with the
            input for rail_fsps
        """

        if seed is not None:
            self.config.seed = seed

        self.set_data('input', sample)
        self.run()
        self.finalize()

        return self.get_handle('output')


class GalaxyPopulationGenerator(GalaxyGenerator):
    """
    Generator that creates galaxy populations from library of galaxy_modelling components.
    It requires the package gal_pop_model_components.
    Install gal_pop_model_components with the following command:
    git clone https://github.com/torluca/gal_pop_model_components
    cd gal_pop_model_components
    pip install -e .

    Parameters
    ----------
    min_wavelength : positive float
        The minimum wavelength of the rest-frame SED.
        Wavelength unit assumed to be Angstrom.

    """

    name = 'GalaxyPopulationGenerator'
    config_options = GalaxyGenerator.config_options.copy()
    config_options.update(config_params_file=Param(str, 'params.yml', msg='yaml file containing the parameters'
                                                                          'for the galaxy_modelling components'),
                          sample_luminosity_function=Param(bool, True, msg='True/False for joint sampling M,z from'
                                                                           'luminosity function'),
                          luminosity_function_type=Param(str, 'Schechter', msg='ignored if '
                                                                               'sample_luminosity_function=False'))
    # but then the actual parameters needed, like input redshift grid, mstar, sky area etc are provided via a configuration file

    def __init__(self, args, comm=None):
        """
        Parameters
        ----------
        """
        GalaxyGenerator.__init__(self, args, comm=comm)
        # validate parameters
        allowed_modelling_components = gal_pop_model_components.list_available_components()
        if self.config.luminosity_function_type not in allowed_modelling_components:
            raise KeyError("The {} component is not implemented in the library"
                           .format(self.config.luminosity_function_type))
        # some check if conditions

    def _sample_galaxy_population_properties(self):
        """
        Parameters
        ----------
        """
        # perhaps first transform self.config into a dictionary
        # self.modelling_components_dictionary = dict(self.config)
        # galaxy_population_properties = gal_pop_model_components.create_galaxy_population(self.config)
        galaxy_population = galaxy_population_modelling.GalaxyPopulationForRail(self.config)
        galaxy_population.update_joint_pdf_galaxy_properties()
        galaxy_population.sample_galaxies_from_distribution()

        return galaxy_population.galaxy_catalog

    # def _create_table_with_properties(self, galaxy_population_properties):
    #     """
    #     Parameters
    #     ----------
    #     """
#
    #     galaxy_population_properties_table = gal_pop_model_components.create_table_from_properties(galaxy_population_properties,
    #                                                                                                self.modelling_components_dictionary)
    #     return galaxy_population_properties_table

    def run(self):
        """
        Run method

        Parameters
        ----------

        """

        galaxy_population_properties_table = self._sample_galaxy_population_properties()

        # galaxy_population_properties_table = self._create_table_with_properties(galaxy_population_properties)

        self.add_data('output', galaxy_population_properties_table)
