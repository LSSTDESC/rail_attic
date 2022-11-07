from rail.creation.engine import Creator
from rail.core.stage import RailStage
from rail.core.data import ModelHandle, FitsHandle
from ceci.config import StageParameter as Param
import numpy as np
from astropy.table import Table
from dsps.flat_wcdm import PLANCK15
from jax import vmap
from jax import jit as jjit
from dsps.photometry_kernels import _calc_rest_mag
from dsps.photometry_kernels import _calc_obs_mag


class DSPSPhotometryCreator(Creator):
    """Base class for Creators that generate synthetic photometric data from a model.

        `Creator` will output a table of photometric data.  The details
        will depend on the particular engine.
    """

    name = "DSPS Photometry Creator"
    config_options = RailStage.config_options.copy()
    config_options.update(filter_data=Param(str, 'lsst_filters.npy', msg='npy file containing the structured numpy '
                                                                         'array of the survey filter wavelengths and'
                                                                         'transmissions'),
                          rest_frame_sed_models=Param(str, 'sed_models.pkl', msg=''),
                          rest_frame_wavelengths=Param(str, 'rest_frame_wave.npy',
                                                       msg='npy file containing the wavelength array'
                                                           'of the rest-frame model seds with'
                                                           'shape (n_wavelength_points)'),
                          galaxy_redshifts=Param(str, 'galaxy_redshifts.npy', msg=''),
                          Om0=Param(float, 0.3, msg=''), Ode0=Param(float, 0.7, msg=''), w0=Param(float, -1, msg=''),
                          wa=Param(float, 0, msg=''), h=Param(float, 0.7, msg=''),
                          use_planck_cosmology=Param(bool, False, msg=''),
                          n_galaxies=int, seed=12345)

    inputs = [("model", ModelHandle)]
    outputs = [("output", FitsHandle)]

    def __init__(self, args, comm=None):
        """Initialize Creator"""
        RailStage.__init__(self, args, comm=comm)
        # self.model = self.config.rest_frame_sed_models
        self._b = [None, 0, 0, 0]
        self._calc_rest_mag_vmap = jjit(vmap(_calc_rest_mag, in_axes=self._b))
        self._c = [None, 0, 0, 0, 0, *[None] * 5]
        self._calc_obs_mag_vmap = jjit(vmap(_calc_obs_mag, in_axes=self._c))
        self.filter_data = np.load(self.config.filter_data)
        self.filter_names = np.array([key for key in self.filter_data.dtype.fields
                                      if 'wave' in key])
        self.filter_wavelengths = np.array([self.filter_data[key] for key in self.filter_data.dtype.fields
                                            if 'wave' in key])
        self.filter_transmissions = np.array([self.filter_data[key] for key in self.filter_data.dtype.fields
                                              if 'trans' in key])
        self.rest_frame_wavelengths = np.load(self.config.rest_frame_wavelengths)
        self.galaxy_redshifts = np.load(self.config.galaxy_redshifts)
        if self.config.use_planck_cosmology:
            self.config.Om0, self.config.Ode0, self.config.w0, self.config.wa, self.config.h = PLANCK15
        if not isinstance(args, dict):  # pragma: no cove
            args = vars(args)
        self.open_model(**args)

    def open_model(self, **kwargs):
        """Load the mode and/or attach it to this Creator

        Keywords
        --------
        model : `object`, `str` or `ModelHandle`
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
        if isinstance(model, ModelHandle):  # pragma: no cover
            if model.has_path:
                self.config["model"] = model.path
        self.model = self.set_data("model", model)
        return self.model

    def sample(self, seed: int = None, **kwargs):
        """Draw samples from the model specified in the configuration.

        This is a method for running a Creator in interactive mode.
        In pipeline mode, the subclass `run` method will be called by itself.

        Parameters
        ----------
        seed: int
            The random seed to control sampling

        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame of the samples

        Notes
        -----
        This method puts  `seed` into the stage configuration
        data, which makes them available to other methods.
        It then calls the `run` method, which must be defined by a subclass.
        Finally, the `DataHandle` associated to the `output` tag is returned.
        """
        self.config["seed"] = seed
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        return self.get_handle("output")

    def run(self):
        """

        """

        filter_wavelengths = np.stack((self.filter_wavelengths,) * self.config.n_galaxies, axis=0)
        filter_transmissions = np.stack((self.filter_transmissions,) * self.config.n_galaxies, axis=0)

        args = (self.rest_frame_wavelengths,
                self.model.reshape((self.config.n_galaxies, len(self.rest_frame_wavelengths))),
                filter_wavelengths, filter_transmissions)

        rest_frame_absolute_mags = self._calc_rest_mag_vmap(*args).reshape((self.config.n_galaxies,
                                                                            len(self.filter_wavelengths)))

        args = (self.rest_frame_wavelengths,
                self.model.reshape((self.config.n_galaxies, len(self.rest_frame_wavelengths))),
                filter_wavelengths, filter_transmissions, self.galaxy_redshifts,
                self.config.Om0, self.config.Ode0, self.config.w0, self.config.wa, self.config.h)

        apparent_magnitudes = self._calc_obs_mag_vmap(*args).reshape((self.config.n_galaxies,
                                                                      len(self.filter_wavelengths)))

        idxs = np.arange(1, self.config.n_galaxies + 1, 1, dtype=int)

        output_table = Table([idxs, rest_frame_absolute_mags, apparent_magnitudes], names=('id', 'abs_mags',
                                                                                           'app_mags'))
        self.add_data('output', output_table)
