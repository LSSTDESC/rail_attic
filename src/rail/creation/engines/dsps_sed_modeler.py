import os
from rail.creation.engine import Modeler
from rail.core.stage import RailStage
from rail.core.utils import RAILDIR
from rail.core.data import Hdf5Handle
from ceci.config import StageParameter as Param
import numpy as np
from jax import vmap
from jax import jit as jjit
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY
from dsps import load_ssp_templates
from dsps import calc_rest_sed_sfh_table_lognormal_mdf
from dsps import calc_rest_sed_sfh_table_met_table


class DSPSSingleSedModeler(Modeler):
    r"""
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

    name = "DSPSSingleSedModeler"
    default_files_folder = os.path.join(RAILDIR, 'rail', 'examples_data', 'creation_data', 'data', 'dsps_default_data')
    config_options = RailStage.config_options.copy()
    config_options.update(ssp_templates_file=Param(str, os.path.join(default_files_folder,
                                                                     'ssp_data_fsps_v3.2_lgmet_age.h5'),
                                                   msg='hdf5 file storing the SSP libraries used to create SEDs'),
                          redshift_key=Param(str, 'redshifts',
                                             msg='Redshift keyword name of the hdf5 dataset containing input galaxy '
                                                 'properties'),
                          cosmic_time_grid_key=Param(str, 'cosmic_time_grid',
                                                     msg='Cosmic time grid keyword name of the hdf5 dataset containing '
                                                         'input galaxy properties, this is the grid over '
                                                         'which the stellar mass build-up takes place in units of Gyr'),
                          star_formation_history_key=Param(str, 'star_formation_history',
                                                           msg='Star-formation history keyword name of the hdf5 '
                                                               'dataset containing input galaxy properties, this is '
                                                               'the star-formation history of the galaxy in units of '
                                                               'Msun/yr'),
                          stellar_metallicity_key=Param(str, 'stellar_metallicity',
                                                        msg='Stellar metallicity keyword name of the hdf5 dataset '
                                                            'containing input galaxy properties, this is the stellar'
                                                            ' metallicity in units of log10(Z/Zsun)'),
                          stellar_metallicity_scatter_key=Param(str, 'stellar_metallicity_scatter',
                                                                msg='Stellar metallicity scatter keyword name of the '
                                                                    'hdf5 dataset containing input galaxy properties, '
                                                                    'this is lognormal scatter in the metallicity '
                                                                    'distribution function'),
                          restframe_sed_key=Param(str, 'restframe_seds', msg='Rest-frame SED keyword name of the '
                                                                             'output hdf5 dataset'))

    inputs = [("input", Hdf5Handle)]
    outputs = [("model", Hdf5Handle)]

    def __init__(self, args, comm=None):
        """
        Initialize Modeler

        Parameters
        ----------
        args:
        comm:

        """
        RailStage.__init__(self, args, comm=comm)

        if not os.path.isfile(self.config.ssp_templates_file):
            default_files_folder = os.path.join(RAILDIR, 'rail', 'examples_data', 'creation_data', 'data',
                                                'dsps_default_data')
            os.system('curl -O https://portal.nersc.gov/cfs/lsst/schmidt9/ssp_data_fsps_v3.2_lgmet_age.h5 '
                      '--output-dir {}'.format(default_files_folder))

    def _get_rest_frame_seds(self, ssp_data, redshifts, cosmic_time_grids, star_formation_histories,
                             stellar_metallicities, stellar_metallicities_scatter):
        """
        Computes the rest-frame SED with DSPS based on user-supplied input galaxy population properties.
        The functions calc_rest_sed_sfh_table_lognormal_mdf and calc_rest_sed_sfh_table_met_table
        return a RestSED object composed of
        rest_sedndarray of shape (n_wave, )
            Restframe SED of the galaxy in units of Lsun/Hz
        weightsndarray of shape (n_met, n_ages, 1)
            SSP weights of the joint distribution of stellar age and metallicity
        lgmet_weightsndarray of shape (n_met, )
            SSP weights of the distribution of stellar metallicity
        age_weightsndarray of shape (n_ages, )
            SSP weights of the distribution of stellar age

        Parameters
        ----------
        ssp_data
        redshifts
        cosmic_time_grids
        star_formation_histories
        stellar_metallicities
        stellar_metallicities_scatter

        Returns
        -------

        """

        restframe_seds = {}

        for i in self.split_tasks_by_rank(range(len(redshifts))):
            t_obs = age_at_z(redshifts[i], *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs
            t_obs = t_obs[0]

            if np.isscalar(stellar_metallicities[i]):
                restframe_sed = calc_rest_sed_sfh_table_lognormal_mdf(cosmic_time_grids[i], star_formation_histories[i],
                                                                      stellar_metallicities[i],
                                                                      stellar_metallicities_scatter[i],
                                                                      ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr,
                                                                      ssp_data.ssp_flux, t_obs)
            elif len(stellar_metallicities[i]) > 1:
                restframe_sed = calc_rest_sed_sfh_table_met_table(cosmic_time_grids[i], star_formation_histories[i],
                                                                  stellar_metallicities[i],
                                                                  stellar_metallicities_scatter[i], ssp_data.ssp_lgmet,
                                                                  ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs)
            else:
                raise ValueError

            restframe_seds[i] = restframe_sed.rest_sed

        if self.comm is not None:  # pragma: no cover
            restframe_seds = self.comm.gather(restframe_seds)

            if self.rank != 0:  # pragma: no cover
                return None, None

            restframe_seds = {k: v for a in restframe_seds for k, v in a.items()}

        restframe_seds = np.array([restframe_seds[i] for i in range(len(redshifts))])

        return restframe_seds

    def fit_model(self):
        """
        Produce a creation model from which photometry can be generated

        Parameters
        ----------
        [The parameters depend entirely on the modeling approach!]

        Returns
        -------
        model: ModelHandle
            ModelHandle storing the rest-frame SED model
        """
        self.run()
        self.finalize()
        model = self.get_handle("model")
        return model

    def run(self):
        """
        Run method. It Calls `_get_rest_frame_seds` from DSPS to create a galaxy rest-frame SED.

        Notes
        -----
        The initial stellar mass of the galaxy is 0.
        The definition of the stellar mass table as cumulative sum refers to the total stellar mass formed.
        DSPS conveniently provides IMF-dependent fitting functions to compute the surviving mass
        (see surviving_mstar.py).
        The units of the resulting rest-frame SED is solar luminosity per Hertz. The luminosity refers to that
        emitted by the formed mass at the time of observation.

        Returns
        -------

        """
        input_galaxy_properties = self.get_data('input')
        ssp_data = load_ssp_templates(fn=self.config.ssp_templates_file)

        redshifts = input_galaxy_properties[self.config.redshift_key][()]
        cosmic_time_grids = input_galaxy_properties[self.config.cosmic_time_grid_key][()]
        star_formation_histories = input_galaxy_properties[self.config.star_formation_history_key][()]
        stellar_metallicities = input_galaxy_properties[self.config.stellar_metallicity_key][()]
        stellar_metallicities_scatter = input_galaxy_properties[self.config.stellar_metallicity_scatter_key][()]

        restframe_seds = self._get_rest_frame_seds(ssp_data, redshifts, cosmic_time_grids, star_formation_histories,
                                                   stellar_metallicities, stellar_metallicities_scatter)

        if self.rank == 0:
            rest_frame_sed_models = {self.config.restframe_sed_key: restframe_seds,
                                     self.config.redshift_key: redshifts}
            self.add_data('model', rest_frame_sed_models)


class DSPSPopulationSedModeler(Modeler):
    r"""
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

    name = "DSPSPopulationSedModeler"
    default_files_folder = os.path.join(RAILDIR, 'rail', 'examples_data', 'creation_data', 'data', 'dsps_default_data')
    config_options = RailStage.config_options.copy()
    config_options.update(ssp_templates_file=Param(str, os.path.join(default_files_folder,
                                                                     'ssp_data_fsps_v3.2_lgmet_age.h5'),
                                                   msg='hdf5 file storing the SSP libraries used to create SEDs'),
                          redshift_key=Param(str, 'redshift',
                                             msg='Redshift keyword name of the hdf5 dataset containing input galaxy '
                                                 'properties'),
                          cosmic_time_grid_key=Param(str, 'cosmic_time_grid',
                                                     msg='Cosmic time grid keyword name of the hdf5 dataset containing '
                                                         'input galaxy properties, this is the grid over '
                                                         'which the stellar mass build-up takes place in units of Gyr'),
                          star_formation_history_key=Param(str, 'star_formation_history',
                                                           msg='Star-formation history keyword name of the hdf5 '
                                                               'dataset containing input galaxy properties, this is '
                                                               'the star-formation history of the galaxy in units of '
                                                               'Msun/yr'),
                          stellar_metallicity_key=Param(str, 'stellar_metallicity',
                                                        msg='Stellar metallicity keyword name of the hdf5 dataset '
                                                            'containing input galaxy properties, this is the stellar'
                                                            ' metallicity in units of log10(Z/Zsun)'),
                          stellar_metallicity_scatter_key=Param(str, 'stellar_metallicity_scatter',
                                                                msg='Stellar metallicity scatter keyword name of the '
                                                                    'hdf5 dataset containing input galaxy properties, '
                                                                    'this is lognormal scatter in the metallicity '
                                                                    'distribution function'),
                          restframe_sed_key=Param(str, 'restframe_seds', msg='Rest-frame SED keyword name of the '
                                                                             'output hdf5 dataset'))

    inputs = [("input", Hdf5Handle)]
    outputs = [("model", Hdf5Handle)]

    def __init__(self, args, comm=None):
        r"""
        Initialize Modeler.
        The _a tuple for jax is composed of None or 0, depending on whether you don't or do want the
        array axis to map over for all arguments.

        Parameters
        ----------
        args:
        comm:
        """

        RailStage.__init__(self, args, comm=comm)

        if not os.path.isfile(self.config.ssp_templates_file):
            default_files_folder = os.path.join(RAILDIR, 'rail', 'examples_data', 'creation_data', 'data',
                                                'dsps_default_data')
            os.system('curl -O https://portal.nersc.gov/cfs/lsst/schmidt9/ssp_data_fsps_v3.2_lgmet_age.h5 '
                      '--output-dir {}'.format(default_files_folder))

    def _get_rest_frame_seds(self, ssp_data, redshifts, cosmic_time_grids, star_formation_histories,
                             stellar_metallicities, stellar_metallicities_scatter):
        """
        Computes the rest-frame SEDs with DSPS vmap based on user-supplied input galaxy population properties.

        Parameters
        ----------
        ssp_data
        redshifts
        cosmic_time_grids
        star_formation_histories
        stellar_metallicities
        stellar_metallicities_scatter

        Returns
        -------

        """

        # consider the whole chunk
        self._a = (0, 0, 0, 0, None, None, None, 0)

        if np.isscalar(stellar_metallicities[0]):
            self._calc_sed_vmap = jjit(vmap(calc_rest_sed_sfh_table_lognormal_mdf, in_axes=self._a))
        elif len(stellar_metallicities[0]) > 1:
            self._calc_sed_vmap = jjit(vmap(calc_rest_sed_sfh_table_met_table, in_axes=self._a))
        else:
            raise ValueError

        self._b = (0, None, None, None, None)
        self._calc_age_at_z_vmap = jjit(vmap(age_at_z, in_axes=self._b))
        args_pop_z = (redshifts, *DEFAULT_COSMOLOGY)
        t_obs = self._calc_age_at_z_vmap(*args_pop_z)[:, 0]

        args_pop = (cosmic_time_grids, star_formation_histories, stellar_metallicities,
                    stellar_metallicities_scatter, ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux,
                    t_obs)

        restframe_seds_galpop = self._calc_sed_vmap(*args_pop)

        return restframe_seds_galpop.rest_sed

    def fit_model(self):
        """
        Produce a creation model from which photometry can be generated

        Parameters
        ----------
        [The parameters depend entirely on the modeling approach!]

        Returns
        -------
        model: ModelHandle
            ModelHandle storing the rest-frame SED models
        """

        self.run()
        self.finalize()
        model = self.get_handle("model")
        return model

    def run(self):
        """
        Run method

        Calls `_get_rest_frame_seds` from DSPS to create galaxy rest-frame SEDs for a galaxy population.

        Notes
        -----
        The definition of the stellar mass table as cumulative sum refers to the total stellar mass formed.
        DSPS conveniently provides IMF-dependent fitting functions to compute the surviving mass
        (see surviving_mstar.py).
        The units of the resulting rest-frame SEDs are solar luminosity per Hertz. The luminosity refers to that
        emitted by the formed mass at the time of observation.

        Returns
        -------

        """

        input_galaxy_properties = self.get_data('input')
        ssp_data = load_ssp_templates(fn=self.config.ssp_templates_file)

        redshifts = input_galaxy_properties[self.config.redshift_key][()]
        cosmic_time_grids = input_galaxy_properties[self.config.cosmic_time_grid_key][()]
        star_formation_histories = input_galaxy_properties[self.config.star_formation_history_key][()]
        stellar_metallicities = input_galaxy_properties[self.config.stellar_metallicity_key][()]
        stellar_metallicities_scatter = input_galaxy_properties[self.config.stellar_metallicity_scatter_key][()]

        restframe_seds = self._get_rest_frame_seds(ssp_data, redshifts, cosmic_time_grids, star_formation_histories,
                                                   stellar_metallicities, stellar_metallicities_scatter)

        rest_frame_sed_models = {self.config.restframe_sed_key: restframe_seds,
                                 self.config.redshift_key: redshifts}
        self.add_data('model', rest_frame_sed_models)
