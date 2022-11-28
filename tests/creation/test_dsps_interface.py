import os
import subprocess
import pytest
from src.rail.creation.galaxy_modelling.dsps_sed_modeler import DSPSSingleSedModeler, DSPSPopulationSedModeler
from rail.core.stage import RailStage
from rail.core.utils import RAILDIR


default_files_folder = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'dsps_data')


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
        ({"stellar_mass_type": 'tabulated'}, KeyError),
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

    with pytest.raises(error):
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        single_sed_model = DSPSSingleSedModeler.make_stage(name='DSPSsingleSEDmodel',
                                                           galaxy_age=7, galaxy_metallicity=0.0,
                                                           galaxy_metallicity_scatter=0.2, **settings)
        single_sed_model.fit_model()


def test_DSPSSingleSedModeler_model_creation():
    """
    Test if the resulting ModelHandle is not empty.

    Returns
    -------

    """
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    single_sed_model = DSPSSingleSedModeler.make_stage(name='DSPSsingleSEDmodel',
                                                       galaxy_age=7, galaxy_metallicity=0.0,
                                                       galaxy_metallicity_scatter=0.2)
    model_handle = single_sed_model.fit_model()
    subprocess.run(['rm', 'model_DSPSsingleSEDmodel.pkl'])
    assert bool(model_handle) is True


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"galaxy_age": os.path.join(default_files_folder, 'galaxy_population_wrong_ages.npy')}, ValueError),
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

    with pytest.raises(error):
        DSPSPopulationSedModeler.make_stage(**settings)


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"galaxy_metallicity": os.path.join(default_files_folder,
                                             'galaxy_population_wrong_metallicities.npy')}, ValueError),
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

    with pytest.raises(error):
        DSPSPopulationSedModeler.make_stage(**settings)


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"stellar_mass_type": 'tabulated'}, KeyError),
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

    with pytest.raises(error):
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        population_seds_model = DSPSPopulationSedModeler.make_stage(name='DSPSPopulationSEDmodel', **settings)
        population_seds_model.fit_model()


def test_DSPSPopulationSedModeler_model_creation():
    """
    Test if the resulting ModelHandle is not empty.

    Returns
    -------

    """
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    population_seds_model = DSPSPopulationSedModeler.make_stage(name='DSPSPopulationSEDmodel')
    model_handle = population_seds_model.fit_model()
    subprocess.run(['rm', 'model_DSPSPopulationSEDmodel.pkl'])
    assert bool(model_handle) is True
