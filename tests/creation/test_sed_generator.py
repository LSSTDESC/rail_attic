import os
os.environ["SPS_HOME"] = "/opt/hostedtoolcache/Python/fsps"

import tables_io
from rail.creation.sed_generation.sed_generator import FSPSSedGenerator
import pytest
from rail.core.stage import RailStage


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"min_wavelength": -1}, ValueError),
        ({"max_wavelength": -1}, ValueError),
    ],
)
def test_FSPSSedGenerator_bad_wavelength_range(settings, error):
    """Test bad wavelength range that should raise Value and Type errors."""
    with pytest.raises(error):
        FSPSSedGenerator.make_stage(name='sed_generator_test', **settings)


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"sfh_type": 3}, AssertionError),
    ],
)
def test_FSPSSedGenerator_bad_tabulated_sfh_params(settings, error):
    """Test bad tabulated sfh params that should raise Value and Type errors."""
    with pytest.raises(error):
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        dummy_io_data = tables_io.read('tests/data/test_fsps_sed.fits')
        sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', zcontinuous=1,
                                                          add_neb_emission=True, physical_units=True, **settings)
        sed_generation_test.add_data('input', dummy_io_data)
        sed_generation_test.run()


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"smooth_lsf": True}, AssertionError),
    ],
)
def test_FSPSSedGenerator_bad_tabulated_lsf_params(settings, error):
    """Test bad tabulated sfh params that should raise Value and Type errors."""
    with pytest.raises(error):
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        dummy_io_data = tables_io.read('tests/data/test_fsps_sed.fits')
        sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', smooth_velocity=False,
                                                          physical_units=True, **settings)
        sed_generation_test.add_data('input', dummy_io_data)
        sed_generation_test.run()


def test_FSPSSedGenerator_output_table():
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    dummy_io_data = tables_io.read('tests/data/test_fsps_sed.fits')
    sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', physical_units=True,)
    sed_generation_test.add_data('input', dummy_io_data)
    sed_generation_test.run()
    out_table = sed_generation_test.get_data('output')

    assert 'wavelength' in out_table.colnames
    assert 'spectrum' in out_table.colnames
