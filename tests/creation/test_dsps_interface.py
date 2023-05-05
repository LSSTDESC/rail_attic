import os
import subprocess

import numpy as np
import pytest
from dsps.utils import _jax_get_dt_array

from rail.core.stage import RailStage
from rail.core.utils import RAILDIR
from src.rail.creation.engines.dsps_photometry_creator import DSPSPhotometryCreator
from src.rail.creation.engines.dsps_sed_modeler import DSPSPopulationSedModeler, DSPSSingleSedModeler

default_files_folder = os.path.join(RAILDIR, "rail", "examples_data", "testdata")



def save_to_npy(filenames, properties):
    """

    Parameters
    ----------
    filenames
    properties

    Returns
    -------

    """

    for i in range(len(filenames)):
        np.save(filenames[i], properties[i])


def create_testdata_DSPSSingleSedModeler(default_files_folder):
    """

    Parameters
    ----------
    default_files_folder

    Returns
    -------

    """

    testdata_filenames = {
        "age_grid": os.path.join(default_files_folder, "age_grid.npy"),
        "metallicity_grid": os.path.join(default_files_folder, "metallicity_grid.npy"),
        "star_formation_history": os.path.join(default_files_folder, "SFH.npy"),
        "cosmic_time_grid": os.path.join(default_files_folder, "cosmic_time_table.npy"),
        "stellar_mass_history": os.path.join(default_files_folder, "stellar_mass_table.npy"),
    }

    log_age_gyr = np.arange(-3.5, 1.20, 0.05, dtype="float64")
    lgZsun_bin_mids = np.array(
        [
            -1.97772361,
            -1.80163235,
            -1.67669361,
            -1.5797836,
            -1.50060235,
            -1.37566361,
            -1.2787536,
            -1.19957235,
            -1.07463362,
            -0.97772361,
            -0.88081359,
            -0.78739191,
            -0.68768899,
            -0.58855752,
            -0.49342377,
            -0.39226288,
            -0.29648237,
            -0.19957235,
            -0.10266234,
            0.0,
            0.10145764,
            0.19836765,
        ]
    )
    n_t = 100
    T0 = 13.8
    t_table = np.linspace(0.1, T0, n_t)
    dt_table = _jax_get_dt_array(t_table)
    sfh_table = np.random.uniform(0, 10, t_table.size)
    logsm_table = np.log10(np.cumsum(sfh_table * dt_table)) + 9.0

    save_to_npy(
        list(testdata_filenames.values()), [log_age_gyr, lgZsun_bin_mids, sfh_table, t_table, logsm_table]
    )

    return testdata_filenames


def create_test_data_DSPSPopulationSedModeler(default_files_folder):
    """

    Parameters
    ----------
    default_files_folder

    Returns
    -------

    """

    testdata_filenames = {
        "age_grid": os.path.join(default_files_folder, "age_grid.npy"),
        "metallicity_grid": os.path.join(default_files_folder, "metallicity_grid.npy"),
        "star_formation_histories": os.path.join(default_files_folder, "SFHs.npy"),
        "cosmic_time_grids": os.path.join(default_files_folder, "cosmic_times_table.npy"),
        "stellar_mass_histories": os.path.join(default_files_folder, "stellar_masses_table.npy"),
        "population_ages": os.path.join(default_files_folder, "galaxy_population_ages.npy"),
        "population_metallicities": os.path.join(default_files_folder, "galaxy_population_metallicities.npy"),
        "population_metallicity_scatters": os.path.join(
            default_files_folder, "galaxy_population_metallicity_scatters.npy"
        ),
        "wrong_ages": os.path.join(default_files_folder, "galaxy_population_wrong_ages.npy"),
        "wrong_metallicities": os.path.join(
            default_files_folder, "galaxy_population_wrong_metallicities.npy"
        ),
    }

    log_age_gyr = np.arange(-3.5, 1.20, 0.05, dtype="float64")
    lgZsun_bin_mids = np.array(
        [
            -1.97772361,
            -1.80163235,
            -1.67669361,
            -1.5797836,
            -1.50060235,
            -1.37566361,
            -1.2787536,
            -1.19957235,
            -1.07463362,
            -0.97772361,
            -0.88081359,
            -0.78739191,
            -0.68768899,
            -0.58855752,
            -0.49342377,
            -0.39226288,
            -0.29648237,
            -0.19957235,
            -0.10266234,
            0.0,
            0.10145764,
            0.19836765,
        ]
    )

    n_gal_population = 10
    n_t = 100
    T0 = 13.8
    t_table = np.linspace(0.1, T0, n_t)
    dt_table = _jax_get_dt_array(t_table)
    sfh_table = np.random.uniform(0, 10, t_table.size)
    logsm_table = np.log10(np.cumsum(sfh_table * dt_table)) + 9.0

    sfhs_table = np.empty((n_gal_population, len(sfh_table)))
    for i in range(n_gal_population):
        sfhs_table[i, :] = sfh_table

    ts_table = np.empty((n_gal_population, len(t_table)))
    for i in range(n_gal_population):
        ts_table[i, :] = t_table

    logsms_table = np.empty((n_gal_population, len(logsm_table)))
    for i in range(n_gal_population):
        logsms_table[i, :] = logsm_table

    galaxy_ages = np.random.uniform(low=1, high=13, size=n_gal_population)
    galaxy_metallicities = np.random.choice(lgZsun_bin_mids, size=n_gal_population, replace=False)
    galaxy_metallicity_scatters = np.random.normal(loc=0.2, scale=0.1, size=n_gal_population)

    wrong_ages = np.random.uniform(10, 20, n_gal_population)
    wrong_metallicities = np.random.uniform(-10, -1, n_gal_population)

    save_to_npy(
        list(testdata_filenames.values()),
        [
            log_age_gyr,
            lgZsun_bin_mids,
            sfhs_table,
            ts_table,
            logsms_table,
            galaxy_ages,
            galaxy_metallicities,
            galaxy_metallicity_scatters,
            wrong_ages,
            wrong_metallicities,
        ],
    )

    return testdata_filenames


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"galaxy_age": 15}, ValueError),
    ],
)
def test_DSPSSingleSedModeler_bad_galaxy_age(settings, error):
    """
    Test if galaxy age is in allowed range. If not, it should raise ValueError.

    Parameters
    ----------
    settings: dict
        dictionary having "galaxy_age" as keyword and value outside range to trigger ValueError
    error: built-in type
        ValueError
    Returns
    -------

    """

    with pytest.raises(error):
        DSPSSingleSedModeler.make_stage(**settings)


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"galaxy_metallicity": -3}, ValueError),
    ],
)
def test_DSPSSingleSedModeler_bad_galaxy_metallicity(settings, error):
    """
    Test if galaxy metallicity is in allowed range. If not, it should raise ValueError.

    Parameters
    ----------
    settings: dict
        dictionary having "galaxy_metallicity" as keyword and value outside range to trigger ValueError
    error: built-in type
        ValueError
    Returns
    -------

    """

    with pytest.raises(error):
        DSPSSingleSedModeler.make_stage(**settings)


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"stellar_mass_type": "tabulated"}, KeyError),
    ],
)
def test_DSPSSingleSedModeler_bad_stellar_mass_type(settings, error):
    """
    Test if stellar_mass_type keyword is correct, if not it should raise KeyError.

    Parameters
    ----------
    settings: dict
        dictionary having "stellar_mass_type" as keyword and not implemented key to trigger KeyError
    error: built-in type
        KeyError
    Returns
    -------

    """

    testdata_filenames = create_testdata_DSPSSingleSedModeler(default_files_folder)

    with pytest.raises(error):
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        single_sed_model = DSPSSingleSedModeler.make_stage(
            name="DSPSsingleSEDmodel",
            galaxy_age=7,
            galaxy_metallicity=0.0,
            galaxy_metallicity_scatter=0.2,
            **settings,
        )
        subprocess.run(["rm"] + list(testdata_filenames.values()))
        single_sed_model.fit_model()


def test_DSPSSingleSedModeler_model_creation():
    """
    Test if the resulting ModelHandle is not empty.

    Returns
    -------

    """

    testdata_filenames = create_testdata_DSPSSingleSedModeler(default_files_folder)

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    single_sed_model = DSPSSingleSedModeler.make_stage(
        name="DSPSsingleSEDmodel", galaxy_age=7, galaxy_metallicity=0.0, galaxy_metallicity_scatter=0.2
    )
    model_handle = single_sed_model.fit_model()
    subprocess.run(["rm", "model_DSPSsingleSEDmodel.pkl"])
    subprocess.run(["rm"] + list(testdata_filenames.values()))
    assert bool(model_handle) is True


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"galaxy_age": os.path.join(default_files_folder, "galaxy_population_wrong_ages.npy")}, ValueError),
    ],
)
def test_DSPSPopulationSedModeler_bad_galaxy_ages(settings, error):
    """
    Test if galaxy ages are in the allowed range. If not, it should raise ValueError.

    Parameters
    ----------
    settings: dict
        dictionary having "galaxy_age" as keyword and file path of ages outside range to trigger ValueError
    error: built-in type
        ValueError
    Returns
    -------

    """

    create_test_data_DSPSPopulationSedModeler(default_files_folder)

    with pytest.raises(error):
        DSPSPopulationSedModeler.make_stage(**settings)


@pytest.mark.parametrize(
    "settings,error",
    [
        (
            {
                "galaxy_metallicity": os.path.join(
                    default_files_folder, "galaxy_population_wrong_metallicities.npy"
                )
            },
            ValueError,
        ),
    ],
)
def test_DSPSPopulationSedModeler_bad_galaxy_metallicities(settings, error):
    """
    Test if galaxy metallicities are in the allowed range. If not, it should raise ValueError.

    Parameters
    ----------
    settings: dict
        dictionary having "galaxy_metallicity" as keyword and file path of metallicities outside range
        to trigger ValueError
    error: built-in type
        ValueError
    Returns
    -------

    """

    create_test_data_DSPSPopulationSedModeler(default_files_folder)

    with pytest.raises(error):
        DSPSPopulationSedModeler.make_stage(**settings)


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"stellar_mass_type": "tabulated"}, KeyError),
    ],
)
def test_DSPSPopulationSedModeler_bad_stellar_mass_type(settings, error):
    """
    Test if stellar_mass_type keyword is correct, if not it should raise KeyError.

    Parameters
    ----------
    settings: dict
        dictionary having "stellar_mass_type" as keyword and not implemented key to trigger KeyError
    error: built-in type
        KeyError
    Returns
    -------

    """

    testdata_filenames = create_test_data_DSPSPopulationSedModeler(default_files_folder)

    with pytest.raises(error):
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        population_seds_model = DSPSPopulationSedModeler.make_stage(
            name="model_DSPSPopulationSEDmodel", **settings
        )
        subprocess.run(["rm"] + list(testdata_filenames.values()))
        population_seds_model.fit_model()


def test_DSPSPopulationSedModeler_model_creation():
    """
    Test if the resulting ModelHandle is not empty.

    Returns
    -------

    """

    testdata_filenames = create_test_data_DSPSPopulationSedModeler(default_files_folder)

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    population_seds_model = DSPSPopulationSedModeler.make_stage(name="model_DSPSPopulationSEDmodel")
    model_handle = population_seds_model.fit_model()
    subprocess.run(["rm", "model_DSPSPopulationSEDmodel.pkl"])
    subprocess.run(["rm"] + list(testdata_filenames.values()))
    assert bool(model_handle) is True


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"Om0": 2}, ValueError),
    ],
)
def test_DSPSPhotometryCreator_bad_omega_matter(settings, error):
    """
    Test if omega matter is in allowed range. If not, it should raise ValueError.

    Parameters
    ----------
    settings: dict
        dictionary having "Om0" as keyword and value outside range to trigger ValueError
    error: built-in type
        ValueError
    Returns
    -------

    """

    with pytest.raises(error):
        DSPSPhotometryCreator.make_stage(**settings)


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"Ode0": 2}, ValueError),
    ],
)
def test_DSPSPhotometryCreator_bad_omega_de(settings, error):
    """
    Test if dark energy is in allowed range. If not, it should raise ValueError.

    Parameters
    ----------
    settings: dict
        dictionary having "Ode0" as keyword and value outside range to trigger ValueError
    error: built-in type
        ValueError
    Returns
    -------

    """

    with pytest.raises(error):
        DSPSPhotometryCreator.make_stage(**settings)


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"h": 1.1}, ValueError),
    ],
)
def test_DSPSPhotometryCreator_bad_little_h(settings, error):
    """
    Test if the dimensionless hubble constant is in allowed range. If not, it should raise ValueError.

    Parameters
    ----------
    settings: dict
        dictionary having "h" as keyword and value outside range to trigger ValueError
    error: built-in type
        ValueError
    Returns
    -------

    """

    with pytest.raises(error):
        DSPSPhotometryCreator.make_stage(**settings)
