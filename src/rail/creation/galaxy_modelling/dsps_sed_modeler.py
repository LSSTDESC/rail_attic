from rail.creation.engine import Modeler
from rail.core.stage import RailStage
from rail.core.data import ModelHandle
from ceci.config import StageParameter as Param
import numpy as np
from dsps.seds_from_tables import _calc_sed_kern
from dsps.utils import _jax_get_dt_array
from jax import vmap
from jax import jit as jjit


class DSPSSingleSedModeler(Modeler):
    """
    Derived class of Modeler for creating a single galaxy rest-frame SED model using DSPS (Hearin+21).
    SPS calculations are based on a set of template SEDs of simple stellar populations (SSPs).
    Supplying such templates is outside the planned scope of the DSPS package, and so they
    will need to be retrieved from some other library. For example, the FSPS library supplies
    such templates in a convenient form.

    Notes
    -----
    The user-provided metallicity grid should be consistently defined with the metallicity of the templates SEDs.
    Users should be cautious in the use of the cosmic time grid. The time resolution strongly depends on the
    user scientific aim.

    """

    name = "DSPS single SED model"
    config_options = RailStage.config_options.copy()
    config_options.update(age_grid=Param(str, 'age_grid.npy', msg='npy file containing the age grid values '
                                                                  'in units of log10(Age[Gyr])'),
                          metallicity_grid=Param(str, 'metallicity_grid.npy', msg='npy file containing the metallicity '
                                                                                  'grid values in units of '
                                                                                  'log10(Z / Z_solar)'),
                          ssp_fluxes=Param(str, 'ssp_spec_flux_lines.npy', msg='npy file containing the SSPs template'
                                                                               'SEDs with shape '
                                                                               '(n_grid_metallicity_values,'
                                                                               'n_grid_age_values, '
                                                                               'n_wavelength_points)'),
                          star_formation_history=Param(str, 'SFH.npy', msg='npy file containing star-formation'
                                                                           'history of the galaxy in units of'
                                                                           'solar masses per year'),
                          cosmic_time_grid=Param(str, 'cosmic_time_table.npy', msg='Cosmic time table over which'
                                                                                   'the stellar mass build-up takes'
                                                                                   'place'),
                          stellar_mass_type=Param(str, 'formed', msg='Options are "formed" or "surviving" for the '
                                                                     'computation of stellar-mass build-up'),
                          stellar_mass_table=Param(str, 'stellar_mass_table.npy', msg='npy file containing the'
                                                                                      'log galaxy stellar mass in units'
                                                                                      'of solar masses as function of'
                                                                                      'cosmic time, valid only when'
                                                                                      'stellar_mass_type="surviving"'),
                          galaxy_age=Param(float, 13.0, msg='Galaxy age at the time of observation in Gyr'),
                          galaxy_metallicity=Param(float, 0.0, msg='Galaxy metallicity at the time of observation'
                                                                   'in units of log10(Z / Z_solar)'),
                          galaxy_metallicity_scatter=Param(float, 0.2, msg='Log-normal scatter of the metallicity '
                                                                           'at the time of observation'),
                          seed=12345)

    # inputs = [("input", DataHandle)]
    # outputs = [("model", ModelHandle)]
    outputs = [("model", ModelHandle)]

    def __init__(self, args, comm=None):
        """Initialize Modeler"""
        RailStage.__init__(self, args, comm=comm)
        self.model = None
        self.log_age_gyr = np.load(self.config.age_grid)
        self.lgZsun_bin = np.load(self.config.metallicity_grid)
        self.ssp_flux = np.load(self.config.ssp_fluxes)
        self.sfh_table = np.load(self.config.star_formation_history)
        self.t_table = np.load(self.config.cosmic_time_grid)

    def fit_model(self):
        """
        Produce a creation model from which photometry can be generated

        Parameters
        ----------
        [The parameters depend entirely on the modeling approach!]

        Returns
        -------
        [This will definitely be a file, but the filetype and format depend entirely on the modeling approach!]
        """
        self.run()
        self.finalize()
        return self.get_handle("model")

    def run(self):
        """Run method

        Calls `_calc_sed_kern` from DSPS to create a galaxy rest-frame SED.

        Notes
        -----
        Puts the rest-frame SED into the data store under this stages 'model' tag.
        9.0 in the logsm_table definition is a Gyr to yr conversion factor.
        The initial stellar mass of the galaxy is 0.
        The definition of the stellar mass table as cumulative sum refers to the total stellar mass formed.
        If user requires surviving mass, then logsm_table should be a user-provided npy file.
        DSPS conveniently provides IMF-dependent fitting functions to compute the surviving mass
        (see surviving_mstar.py).
        The units of the resulting rest-frame SED is solar luminosity per Hertz. The luminosity refers to that
        emitted by the formed mass at the time of observation.
        """

        dt_table = _jax_get_dt_array(self.t_table)
        if self.config.stellar_mass_type == 'formed':
            logsm_table = np.log10(np.cumsum(self.sfh_table * dt_table)) + 9.0
        elif self.config.stellar_mass_type == 'surviving':
            logsm_table = np.load(self.config.stellar_mass_table)
        else:
            raise KeyError('Stellar mass type "{}" not implemented'.format(self.config.stellar_mass_type))

        args = (self.config.galaxy_age,
                self.lgZsun_bin,
                self.log_age_gyr,
                self.ssp_flux,
                self.t_table,
                logsm_table,
                self.config.galaxy_metallicity,
                self.config.galaxy_metallicity_scatter)

        restframe_sed = _calc_sed_kern(*args)

        # save the sed model
        self.add_data("model", np.array(restframe_sed))


class DSPSPopulationSedModeler(Modeler):
    """
    Derived class of Modeler for creating a galaxy population rest-frame SED models using DSPS (Hearin+21).
    SPS calculations are based on a set of template SEDs of simple stellar populations (SSPs).
    Supplying such templates is outside the planned scope of the DSPS package, and so they
    will need to be retrieved from some other library. For example, the FSPS library supplies
    such templates in a convenient form.

    Notes
    -----
    The user-provided metallicity grid should be consistently defined with the metallicity of the templates SEDs.
    Users should be cautious in the use of the cosmic time grid. The time resolution strongly depends on the
    user scientific aim.
    jax serially execute the computations on CPU on single core, for CPU parallelization you need MPI.
    If GPU is used, jax natively and automatically parallelize the execution.
    """

    name = "DSPS population SED models"
    config_options = RailStage.config_options.copy()
    config_options.update(age_grid=Param(str, 'age_grid.npy', msg='npy file containing the age grid values '
                                                                  'in units of log10(Age[Gyr])'),
                          metallicity_grid=Param(str, 'metallicity_grid.npy', msg='npy file containing the metallicity '
                                                                                  'grid values in units of '
                                                                                  'log10(Z / Z_solar)'),
                          ssp_fluxes=Param(str, 'ssp_spec_flux_lines.npy', msg='npy file containing the SSPs template'
                                                                               'SEDs with shape '
                                                                               '(n_grid_metallicity_values,'
                                                                               'n_grid_age_values, '
                                                                               'n_wavelength_points)'),
                          star_formation_history=Param(str, 'SFHs.npy', msg='npy file containing star-formation'
                                                                            'histories of the individual galaxies in '
                                                                            'the galaxy population in units of'
                                                                            'solar masses per year'),
                          cosmic_time_grid=Param(str, 'cosmic_times_table.npy', msg='Cosmic time tables of the galaxy'
                                                                                    'population over which'
                                                                                    'the stellar mass build-up takes'
                                                                                    'place'),
                          stellar_mass_type=Param(str, 'formed', msg='Options are "formed" or "surviving" for the '
                                                                     'computation of stellar-mass build-up'),
                          stellar_mass_table=Param(str, 'stellar_masses_table.npy',
                                                   msg='npy file containing the log galaxy stellar masses in units'
                                                       'of solar masses as function of cosmic time, valid only when'
                                                       'stellar_mass_type="surviving"'),
                          galaxy_age=Param(str, 'galaxy_population_ages.npy', msg='npy file containing the galaxy ages'
                                                                                  ' at the time of observation in Gyr'),
                          galaxy_metallicity=Param(str, 'galaxy_population_metallicities.npy',
                                                   msg='npy file containing the galaxy metallicities '
                                                       'at the time of observation in units of log10(Z / Z_solar)'),
                          galaxy_metallicity_scatter=Param(str, 'galaxy_population_metallicity_scatters.npy',
                                                           msg='npy file containing the log-normal scatters of the '
                                                               'galaxy metallicities at the time of observation'),
                          seed=12345)

    # inputs = [("input", DataHandle)]
    # outputs = [("model", ModelHandle)]
    outputs = [("model", ModelHandle)]

    def __init__(self, args, comm=None):
        """Initialize Modeler"""
        RailStage.__init__(self, args, comm=comm)
        self.model = None
        self.log_age_gyr = np.load(self.config.age_grid)
        self.lgZsun_bin = np.load(self.config.metallicity_grid)
        self.ssp_flux = np.load(self.config.ssp_fluxes)
        self.sfh_table_pop = np.load(self.config.star_formation_history)
        self.t_table_pop = np.load(self.config.cosmic_time_grid)
        self.galaxy_age_pop = np.load(self.config.galaxy_age)
        self.galaxy_metallicity_pop = np.load(self.config.galaxy_metallicity)
        self.galaxy_metallicity_scatter_pop = np.load(self.config.galaxy_metallicity_scatter)

    def fit_model(self):
        """
        Produce a creation model from which photometry can be generated

        Parameters
        ----------
        [The parameters depend entirely on the modeling approach!]

        Returns
        -------
        [This will definitely be a file, but the filetype and format depend entirely on the modeling approach!]
        """
        self.run()
        self.finalize()
        return self.get_handle("model")

    def run(self):
        """Run method

        Calls `_calc_sed_kern` from DSPS to create galaxy rest-frame SEDs for a galaxy population.

        Notes
        -----
        9.0 in the logsm_table definition is a Gyr to yr conversion factor.
        The initial stellar mass of the galaxies is 0.
        The definition of the stellar mass table as cumulative sum refers to the total stellar mass formed.
        If user requires surviving mass, then logsm_table should be a user-provided npy file.
        DSPS conveniently provides IMF-dependent fitting functions to compute the surviving mass
        (see surviving_mstar.py).
        The units of the resulting rest-frame SEDs are solar luminosity per Hertz. The luminosity refers to that
        emitted by the formed mass at the time of observation.
        The _a tuple for jax is composed of None or 0, depending on whether you don't or do want the
         array axis to map over for all arguments.
        """

        # _a = (*[None] * 5, 0, 0, 0)
        _a = (0, None, None, None, 0, 0, 0, 0)
        _calc_sed_vmap = jjit(vmap(_calc_sed_kern, in_axes=_a))

        if self.config.stellar_mass_type == 'formed':
            dt_table_pop = np.empty_like(self.t_table_pop)
            for i in range(len(self.t_table_pop)):
                dt_table_pop[i, :] = _jax_get_dt_array(self.t_table_pop[i, :])
            logsm_table_pop = np.log10(np.cumsum(self.sfh_table_pop * dt_table_pop, axis=1)) + 9.0
        elif self.config.stellar_mass_type == 'surviving':
            logsm_table_pop = np.load(self.config.stellar_mass_table)
        else:
            raise KeyError('Stellar mass type "{}" not implemented'.format(self.config.stellar_mass_type))

        args_pop = (self.galaxy_age_pop,
                    self.lgZsun_bin,
                    self.log_age_gyr,
                    self.ssp_flux,
                    self.t_table_pop,
                    logsm_table_pop,
                    self.galaxy_metallicity_pop,
                    self.galaxy_metallicity_scatter_pop)

        restframe_sed_galpop = _calc_sed_vmap(*args_pop)

        # save the sed model
        self.add_data("model", np.array(restframe_sed_galpop))
