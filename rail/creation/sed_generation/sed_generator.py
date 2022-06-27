from rail.creation.sed_generation.generator import Generator
import fsps
import numpy as np
from astropy.table import Table
from ceci.config import StageParameter as Param

class FSPSSedGenerator(Generator):
    """
    Generator that creates rest-frame SEDs with FSPS.
    Install FSPS with the following commands:
    git clone --recursive https://github.com/dfm/python-fsps.git
    cd python-fsps
    python -m pip install .

    Example: generator = SedGenerator(min_wavelength=2960,
                                      max_wavelength=10245)

    add ability to have star-formation history tabulated

    Parameters
    ----------
    min_wavelength : positive float
        The minimum wavelength of the rest-frame SED.
        Wavelength unit assumed to be Angstrom.
    max_wavelength : positive float
        The maximum wavelength of the rest-frame SED.
        Wavelength unit assumed to be Angstrom.
    """

    name = 'SedGenerator'
    config_options = Generator.config_options.copy() # I would put minimal set of parameters to make it run
    config_options.update(compute_vega_mags=Param(bool, False, msg='True/False for Vega/AB magnitudes'),
                          vactoair_flag=Param(bool, False, msg='True/False for air/vacuum wavelength'),
                          zcontinuous=Param(int, 1, msg='Flag for interpolation in metallicity of SSP before CSP'),
                          add_neb_emission=Param(bool, False, msg='Turn on/off nebular emission model based on Cloudy'),
                          add_neb_continuum=Param(bool, False, msg='Turn on/off nebular continuum component'),
                          add_stellar_remnants=Param(bool, True, msg='Turn on/off stellar remnants in stellar mass'),
                          compute_light_ages=Param(bool, False, msg='False/True for mass/light-weighted ages'),
                          nebemlineinspec=Param(bool, False, msg='True to include emission line fluxes in spectrum'),
                          smooth_velocity=Param(bool, False, msg='True/False for smoothing in velocity/wavelength space'),
                          imf_type=Param(int, 1, msg='IMF type, see FSPS manual, default Chabrier IMF'),
                          min_wavelength=Param(float, 3000, msg='minimum rest-frame wavelength'),
                          max_wavelength=Param(float, 10000, msg='maximum rest-frame wavelength'),
                          sfh_type=Param(int, 0, msg='star-formation history type, see FSPS manual, default SSP'),
                          dust_type=Param(int, 2, msg='attenuation curve for dust type, see FSPS manual, default Calzetti'),
                          metalname=Param(str, 'logzsol', msg='metallicity column name'),
                          agename=Param(str, 'tage', msg='age column name'),
                          veldispname=Param(str, 'sigma_smooth', msg='velocity dispersion column name'),
                          gasloguname=Param(str, 'gas_logu', msg='log of the gas ionization parameter'),
                          gaslogzname=Param(str, 'gas_logz', msg='log of the gas-phase metallicity'),
                          tauname=Param(str, 'tau', msg='e-folding time for the SFH column name'),
                          fburstname=Param(str, 'fburst', msg='mass fraction formed in instantaneous burst'),
                          tburstname=Param(str, 'tburst', msg='Universe age when the burst occurred'),
                          ebvname=Param(str, 'dust2', msg='attenuation of old stellar light, E(B-V)'),
                          fagnname=Param(str, 'fagn', msg='fraction bolometric luminosity due to AGN'),
                          agntauname=Param(str, 'agn_tau', msg='optical depth of AGN dust torus'))  # unit test for all these values

    def __init__(self, args, comm=None):
        """
        Parameters
        ----------
        """
        Generator.__init__(self, args, comm=comm)
        # validate parameters
        if self.config.min_wavelength < 0:
            raise ValueError("min_wavelength must be positive, not {self.config.min_wavelength}")
        if self.config.max_wavelength < 0:
            raise ValueError("max_wavelength must be positive, not {self.config.max_wavelength}")

    def _get_rest_frame_seds(self, ages, metallicities, velocity_dispersions, gas_ionizations, gas_metallicities,
                             tau_efolding_times, fracs_instantaneous_burst, ages_instantaneous_burst,
                             e_b_v_attenuations, frac_luminosities_agn, opt_depths_agn,
                             tabulated_sfh_file=None):  # add **kwargs
        """
        Parameters
        ----------

        """
        wavelengths = []
        fluxes = []
        for i in range(len(ages)):  # parallelise
            sp = fsps.StellarPopulation(compute_vega_mags=self.config.compute_vega_mags,
                                        vactoair_flag=self.config.vactoair_flag,
                                        zcontinuous=self.config.zcontinuous,
                                        add_neb_emission=self.config.add_neb_emission,
                                        add_neb_continuum=self.config.add_neb_continuum,
                                        add_stellar_remnants=self.config.add_stellar_remnants,
                                        compute_light_ages=self.config.compute_light_ages,
                                        nebemlineinspec=self.config.nebemlineinspec,
                                        smooth_velocity=self.config.smooth_velocity,
                                        zred=0, logzsol=metallicities[i], imf_type=self.config.imf_type,
                                        sigma_smooth=velocity_dispersions[i],
                                        min_wave_smooth=self.config.min_wavelength,
                                        max_wave_smooth=self.config.max_wavelength,
                                        gas_logu=gas_ionizations[i], gas_logz=gas_metallicities[i],
                                        sfh=self.config.sfh_type, tau=tau_efolding_times[i],
                                        tage=ages[i], fburst=fracs_instantaneous_burst[i],
                                        tburst=ages_instantaneous_burst[i],
                                        dust_type=self.config.dust_type, dust2=e_b_v_attenuations[i],
                                        fagn=frac_luminosities_agn[i], agn_tau=opt_depths_agn[i])

            if self.config.sfh_type == 3:
                assert self.config.zcontinuous == 3, 'zcontinous parameter must be set to 3 when using tabular SFHs'
                assert self.config.add_neb_emission == False, \
                    'add_neb_emission must be set to False when using tabular SFHs'
                age_array, sfr_array, metal_array = np.loadtxt(tabulated_sfh_file)
                sp.set_tabular_sfh(age_array, sfr_array, Z=metal_array)

            wavelength, flux_solar_lum_over_angstrom = sp.get_spectrum(tage=ages[i], peraa=True)
            selected_wave_range = np.where((wavelength >= self.config.min_wavelength) &
                                           (wavelength <= self.config.max_wavelength))
            wavelength = wavelength[selected_wave_range]
            flux_solar_lum_over_angstrom = flux_solar_lum_over_angstrom[selected_wave_range]
            wavelengths.append(wavelength)
            fluxes.append(flux_solar_lum_over_angstrom)
            break

        return np.array(wavelengths), np.array(fluxes)

    def run(self):
        """
        Run method

        Generates SED

        Parameters
        ----------

        """

        data = self.get_data('input')

        # get numpy array of magnitudes
        ages = data[self.config.agename]
        metallicities = data[self.config.metalname]
        velocity_dispersions = data[self.config.veldispname]
        gas_ionizations = data[self.config.gasloguname]
        gas_metallicities = data[self.config.gaslogzname]
        tau_efolding_times = data[self.config.tauname]
        fracs_instantaneous_burst = data[self.config.fburstname]
        ages_instantaneous_burst = data[self.config.tburstname]
        e_b_v_attenuations = data[self.config.ebvname]
        frac_luminosities_agn = data[self.config.fagnname]
        opt_depths_agn = data[self.config.agntauname]

        # get observed magnitudes and magnitude errors
        wavelengths, fluxes = self._get_rest_frame_seds(ages, metallicities, velocity_dispersions, gas_ionizations,
                                                        gas_metallicities, tau_efolding_times,
                                                        fracs_instantaneous_burst, ages_instantaneous_burst,
                                                        e_b_v_attenuations, frac_luminosities_agn, opt_depths_agn)

        output_table = Table([wavelengths, fluxes], names=('wavelength', 'spectrum'))

        # convert in TableHandle
        self.add_data('output', output_table)
