import os
import subprocess

import numpy as np
import h5py
import pytest

from rail.core.stage import RailStage
from rail.core.utils import RAILDIR
from src.rail.creation.engines.dsps_photometry_creator import DSPSPhotometryCreator
from src.rail.creation.engines.dsps_sed_modeler import DSPSPopulationSedModeler, DSPSSingleSedModeler

default_files_folder = os.path.join(RAILDIR, 'rail', 'examples_data', 'creation_data', 'data', 'dsps_default_data')


def create_testdata(files_folder):
    """

    Parameters
    ----------
    files_folder

    Returns
    -------
    test_data_filename

    """

    n_galaxies = 10

    redshift = np.arange(0.1, 1.1, 0.1)

    gal_t_table = np.linspace(0.05, 13.8, 100)  # age of the universe in Gyr
    gal_sfr_table = np.random.uniform(0, 10, gal_t_table.size)  # SFR in Msun/yr

    gal_lgmet = -2.0  # log10(Z)
    gal_lgmet_scatter = 0.2  # lognormal scatter in the metallicity distribution function

    cosmic_time_grid = np.full((n_galaxies, len(gal_t_table)), gal_t_table)
    star_formation_history = np.full((n_galaxies, len(gal_sfr_table)), gal_sfr_table)
    stellar_metallicity = np.full(n_galaxies, gal_lgmet)
    stellar_metallicity_scatter = np.full(n_galaxies, gal_lgmet_scatter)

    test_data_filename = os.path.join(files_folder, 'input_galaxy_properties_dsps.h5')

    with h5py.File(test_data_filename, 'w') as h5table:
        h5table.create_dataset(name='redshifts', data=redshift)
        h5table.create_dataset(name='cosmic_time_grid', data=cosmic_time_grid)
        h5table.create_dataset(name='star_formation_history', data=star_formation_history)
        h5table.create_dataset(name='stellar_metallicity', data=stellar_metallicity)
        h5table.create_dataset(name='stellar_metallicity_scatter', data=stellar_metallicity_scatter)

    return test_data_filename


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"filter_folder": default_files_folder}, AssertionError),
    ],
)
def test_DSPSPhotometryCreator_filtersfolder_not_found(settings, error):
    """
    Test if ssp templates filepath exists.

    Parameters
    ----------
    settings: dict
        dictionary having "filter_folder" as keyword and not existing path to trigger OSError
    error: built-in type
        OSError
    Returns
    -------

    """

    with pytest.raises(error):
        DSPSPhotometryCreator.make_stage(**settings)


def test_DSPSSingleSedModeler_model_creation():
    """
    Test if the resulting Hdf5Handle is not empty.

    Returns
    -------

    """

    trainFile = create_testdata(default_files_folder)

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

    single_sed_model = DSPSSingleSedModeler.make_stage(name='DSPS_single_SED_model',
                                                       ssp_templates_file=
                                                       os.path.join(default_files_folder,
                                                                    'ssp_data_fsps_v3.2_lgmet_age.h5'),
                                                       redshift_key='redshifts',
                                                       cosmic_time_grid_key='cosmic_time_grid',
                                                       star_formation_history_key='star_formation_history',
                                                       stellar_metallicity_key='stellar_metallicity',
                                                       stellar_metallicity_scatter_key='stellar_metallicity_scatter',
                                                       restframe_sed_key='restframe_seds')
    h5table = h5py.File(trainFile, 'r')
    single_sed_model.add_data('input', h5table)
    single_sed_model.fit_model()
    h5table.close()

    rest_frame_sed_models = single_sed_model.get_data('model')
    restframe_seds = rest_frame_sed_models['restframe_seds']

    subprocess.run(["rm", "model_DSPS_single_SED_model.hdf5"])
    subprocess.run(["rm", trainFile])
    assert len(restframe_seds) != 0


def test_DSPSPopulationSedModeler_model_creation():
    """
    Test if the resulting Hdf5Handle is not empty.

    Returns
    -------

    """

    trainFile = create_testdata(default_files_folder)

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

    DSPS_population_SED_model = DSPSPopulationSedModeler.make_stage(name='DSPS_population_SED_model',
                                                                    ssp_templates_file=
                                                                    os.path.join(default_files_folder,
                                                                                 'ssp_data_fsps_v3.2_lgmet_age.h5'),
                                                                    redshift_key='redshifts',
                                                                    cosmic_time_grid_key='cosmic_time_grid',
                                                                    star_formation_history_key='star_formation_history',
                                                                    stellar_metallicity_key='stellar_metallicity',
                                                                    stellar_metallicity_scatter_key=
                                                                    'stellar_metallicity_scatter',
                                                                    restframe_sed_key='restframe_seds')

    h5table = h5py.File(trainFile, 'r')
    DSPS_population_SED_model.add_data('input', h5table)
    DSPS_population_SED_model.fit_model()
    h5table.close()

    rest_frame_sed_models = DSPS_population_SED_model.get_data('model')
    restframe_seds = rest_frame_sed_models['restframe_seds']

    subprocess.run(["rm", "model_DSPS_population_SED_model.hdf5"])
    subprocess.run(["rm", trainFile])
    assert len(restframe_seds) != 0


def test_DSPSPhotometryCreator_photometry_creation():
    """
    Test if the resulting Hdf5Handle is not empty.

    Returns
    -------

    """

    trainFile = create_testdata(default_files_folder)

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

    single_sed_model = DSPSSingleSedModeler.make_stage(name='DSPS_single_SED_model',
                                                       ssp_templates_file=
                                                       os.path.join(default_files_folder,
                                                                    'ssp_data_fsps_v3.2_lgmet_age.h5'),
                                                       redshift_key='redshifts',
                                                       cosmic_time_grid_key='cosmic_time_grid',
                                                       star_formation_history_key='star_formation_history',
                                                       stellar_metallicity_key='stellar_metallicity',
                                                       stellar_metallicity_scatter_key='stellar_metallicity_scatter',
                                                       restframe_sed_key='restframe_seds')
    h5table = h5py.File(trainFile, 'r')
    single_sed_model.add_data('input', h5table)
    single_sed_model.fit_model()
    h5table.close()

    trainFile_photometry = 'model_DSPS_single_SED_model.hdf5'

    DSPS_photometry_creator = DSPSPhotometryCreator.make_stage(name='DSPS_photometry_creator',
                                                               redshift_key='redshifts',
                                                               restframe_sed_key='restframe_seds',
                                                               absolute_mags_key='rest_frame_absolute_mags',
                                                               apparent_mags_key='apparent_mags',
                                                               filter_folder=os.path.join(default_files_folder,
                                                                                          'filters'),
                                                               instrument_name='lsst',
                                                               wavebands='u,g,r,i,z',
                                                               ssp_templates_file=
                                                               os.path.join(default_files_folder,
                                                                            'ssp_data_fsps_v3.2_lgmet_age.h5'))

    h5table = h5py.File(trainFile_photometry, 'r')
    DSPS_photometry_creator.add_data('model', h5table)
    output_mags = DSPS_photometry_creator.sample()
    h5table.close()

    subprocess.run(["rm", trainFile_photometry])
    subprocess.run(["rm", trainFile])
    subprocess.run(["rm", 'output_DSPS_photometry_creator.hdf5'])

    assert len(output_mags.data['apparent_mags']) != 0
