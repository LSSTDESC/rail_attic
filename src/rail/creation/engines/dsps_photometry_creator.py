import os
from rail.core.utils import RAILDIR
from rail.creation.engine import Creator
from rail.core.stage import RailStage
from rail.core.data import Hdf5Handle
from ceci.config import StageParameter as Param
import numpy as np
from jax import vmap
from jax import jit as jjit
from dsps import calc_rest_mag
from dsps import calc_obs_mag
from dsps import load_ssp_templates
from dsps import load_transmission_curve
from dsps.cosmology import DEFAULT_COSMOLOGY


class DSPSPhotometryCreator(Creator):
    """
    Derived class of Creator that generate synthetic photometric data from one or more SED models
    generated with the DSPSSingleSedModeler and DSPSPopulationSedModeler classes.
    The user is required to provide files in .npy format for the code to run. Details of what each file should
    contain are explicited in config_options.
    It accepts as input ModelHandles stored in pickle files and as output a Fits table containing magnitudes.

    """

    name = "DSPS Photometry Creator"
    default_files_folder = os.path.join(RAILDIR, 'rail', 'examples_data', 'creation_data', 'data', 'dsps_default_data')
    config_options = RailStage.config_options.copy()
    config_options.update(filter_folder=Param(str, os.path.join(default_files_folder, 'filters'),
                                              msg='Folder containing filter transmissions'),
                          instrument_name=Param(str, 'lsst', msg='Instrument name as prefix to filter transmission'
                                                                 ' files'),
                          wavebands=Param(str, 'u,g,r,i,z,y', msg='Comma-separated list of wavebands'),
                          ssp_templates_file=Param(str, os.path.join(default_files_folder,
                                                                     'ssp_data_fsps_v3.2_lgmet_age.h5'),
                                                   msg='hdf5 file storing the SSP libraries used to create SEDs'))

    inputs = [("model", Hdf5Handle)]
    outputs = [("output", Hdf5Handle)]

    def __init__(self, args, comm=None):
        """
        Initialize class.
        The _b and _c tuples for jax are composed of None or 0, depending on whether you don't or do want the
        array axis to map over for all arguments.
        Parameters
        ----------
        args:
        comm:
        """
        RailStage.__init__(self, args, comm=comm)

        if not os.path.isfile(self.config.ssp_templates_file):
            raise OSError("File {self.config.ssp_templates_file} not found")

        if not os.path.isdir(self.config.filter_folder):
            raise OSError("File {self.config.filter_folder} not found")

        wavebands = self.config.wavebands.split(',')
        self.filter_wavelengths = np.array([load_transmission_curve(fn=os.path.join(self.config.filter_folder,
                                                                                    '{}_{}_transmission.h5'
                                                                                    .format(self.config.instrument_name,
                                                                                            waveband))).wave
                                            for waveband in wavebands])

        self.filter_transmissions = np.array([load_transmission_curve(fn=os.path.join(self.config.filter_folder,
                                                                                      '{}_{}_transmission.h5'
                                                                                      .format(self.config.
                                                                                              instrument_name,
                                                                                              waveband))).transmission
                                             for waveband in wavebands])

        # self.model = self.config.rest_frame_sed_models
        # if self.config.use_planck_cosmology:
        #     self.config.Om0, self.config.Ode0, self.config.w0, self.config.wa, self.config.h = PLANCK15
        # if (self.config.Om0 < 0.) | (self.config.Om0 > 1.):
        #     raise ValueError("The mass density at the current time {self.config.Om0} is outside of allowed"
        #                      " range 0. < Om0 < 1.")
        # if (self.config.Ode0 < 0.) | (self.config.Ode0 > 1.):
        #     raise ValueError("The dark energy density at the current time {self.config.Ode0} is outside of allowed"
        #                      " range 0. < Ode0 < 1.")
        # if (self.config.h < 0.) | (self.config.h > 1.):
        #     raise ValueError("The dimensionless Hubble constant {self.config.h} is outside of allowed"
        #                      " range 0 < h < 1")

        # self._b = [None, 0, 0, 0]
        # self._calc_rest_mag_vmap = jjit(vmap(_calc_rest_mag, in_axes=self._b))
        # self._c = [None, 0, 0, 0, 0, *[None] * 5]
        # self._calc_obs_mag_vmap = jjit(vmap(_calc_obs_mag, in_axes=self._c))
        # self.filter_data = np.load(self.config.filter_data)
        # self.filter_names = np.array([key for key in self.filter_data.dtype.fields
        #                               if 'wave' in key])
        # self.filter_wavelengths = np.array([self.filter_data[key] for key in self.filter_data.dtype.fields
        #                                     if 'wave' in key])
        # self.filter_transmissions = np.array([self.filter_data[key] for key in self.filter_data.dtype.fields
        #                                       if 'trans' in key])
        # self.rest_frame_wavelengths = np.load(self.config.rest_frame_wavelengths)
        # self.galaxy_redshifts = np.load(self.config.galaxy_redshifts)

        if not isinstance(args, dict):  # pragma: no cover
            args = vars(args)
        self.open_model(**args)

    def open_model(self, **kwargs):
        """Load the mode and/or attach it to this Creator

        Keywords
        --------
        model : `object`, `str` or `Hdf5Handle`
            Either an object with a trained model,
            a path pointing to a file that can be read to obtain the trained model,
            or a `ModelHandle` providing access to the trained model.

        Returns
        -------
        self.model : `object`
            The object encapsulating the trained model.
        """

        model = kwargs.get("rest_frame_sed_models", None)
        if model is None or model == "None":  # pragma: no cover
            self.model = None
            return self.model
        if isinstance(model, str):  # pragma: no cover
            self.model = self.set_data("model", data=None, path=model)
            self.config["model"] = model
            return self.model
        if isinstance(model, Hdf5Handle):  # pragma: no cover
            if model.has_path:
                self.config["model"] = model.path
        self.model = self.set_data("model", model)
        return self.model

    def sample(self, seed: int = None, **kwargs):
        r"""
        Creates observed and absolute magnitudes for population of galaxies and stores them into Fits table.

        This is a method for running in interactive mode.
        In pipeline mode, the subclass `run` method will be called by itself.

        Parameters
        ----------
        seed: int
            The random seed to control sampling

        Returns
        -------
        output: astropy.table.Table
            Fits table storing galaxy magnitudes.

        Notes
        -----
        This method puts  `seed` into the stage configuration
        data, which makes them available to other methods.
        It then calls the `run` method.
        Finally, the `FitsHandle` associated to the `output` tag is returned.

        """
        self.config["seed"] = seed
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        output = self.get_handle("output")
        return output

    def _compute_rest_frame_absolute_magnitudes(self, ssp_data, rest_frame_seds, filter_wavelengths,
                                                filter_transmissions):
        """

        Parameters
        ----------
        ssp_data
        rest_frame_seds
        filter_wavelengths
        filter_transmissions

        Returns
        -------

        """
        restframe_abs_mags = {}

        for i in self.split_tasks_by_rank(range(len(rest_frame_seds))):
            self._b = [None, 0, 0, 0]
            self._calc_rest_mag_vmap = jjit(vmap(calc_rest_mag, in_axes=self._b))

            args_abs_mags = (ssp_data.ssp_wave, rest_frame_seds, filter_wavelengths,
                             filter_transmissions)

            restframe_abs_mag = self._calc_rest_mag_vmap(*args_abs_mags)

            restframe_abs_mags[i] = np.array(restframe_abs_mag)

        if self.comm is not None:  # pragma: no cover
            restframe_abs_mags = self.comm.gather(restframe_abs_mags)

            if self.rank != 0:  # pragma: no cover
                return None, None

            restframe_abs_mags = {k: v for a in restframe_abs_mags for k, v in a.items()}

        restframe_abs_mags = np.array([restframe_abs_mags[i] for i in range(len(rest_frame_seds))])

        return restframe_abs_mags

    def _compute_apparent_magnitudes(self, ssp_data, rest_frame_seds, redshifts, filter_wavelengths,
                                     filter_transmissions):
        apparent_mags = {}

        for i in self.split_tasks_by_rank(range(len(redshifts))):
            self._c = [None, 0, 0, 0, 0, None]
            self._calc_app_mag_vmap = jjit(vmap(calc_obs_mag, in_axes=self._c))

            args_app_mags = (ssp_data.ssp_wave, rest_frame_seds, filter_wavelengths,
                             filter_transmissions, redshifts, *DEFAULT_COSMOLOGY)

            apparent_mag = self._calc_app_mag_vmap(*args_app_mags)

            apparent_mags[i] = np.array(apparent_mag)

        if self.comm is not None:  # pragma: no cover
            apparent_mags = self.comm.gather(apparent_mags)

            if self.rank != 0:  # pragma: no cover
                return None, None

            apparent_mags = {k: v for a in apparent_mags for k, v in a.items()}

        apparent_mags = np.array([apparent_mags[i] for i in range(len(redshifts))])

        return apparent_mags

    def run(self):
        """
        This function computes rest-frame absolute magnitudes in the provided wavebands for all the galaxies
        in the population by calling `_calc_rest_mag_vmap` from DSPS. It does the same for the observed
        magnitudes in the AB system by calling `_calc_obs_mag_vmap` from DSPS.
        It then stores both kind of magnitudes and the galaxy indices into an astropy.table.Table.

        Returns
        -------

        """

        ssp_data = load_ssp_templates(fn=self.config.ssp_templates_file)
        filter_wavelengths = np.stack((self.filter_wavelengths,) * self.config.n_galaxies, axis=0)
        filter_transmissions = np.stack((self.filter_transmissions,) * self.config.n_galaxies, axis=0)

        redshifts = self.model['redshifts']
        rest_frame_seds = self.model['restframe_seds']

        rest_frame_absolute_mags = self._compute_rest_frame_absolute_magnitudes(ssp_data, rest_frame_seds,
                                                                                filter_wavelengths,
                                                                                filter_transmissions)
        apparent_mags = self._compute_apparent_magnitudes(ssp_data, rest_frame_seds, redshifts, filter_wavelengths,
                                                          filter_transmissions)

        if self.rank == 0:
            idxs = np.arange(1, len(redshifts) + 1, 1, dtype=int)
            output_mags = {'id': idxs, 'rest_frame_absolute_mags': rest_frame_absolute_mags,
                           'apparent_mags': apparent_mags}  # (n_galaxies, n_wavelengths) = (100000000, 4096)
            self.add_data('output', output_mags)
