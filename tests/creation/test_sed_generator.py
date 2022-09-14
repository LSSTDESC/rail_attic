import os
if "SPS_HOME" not in os.environ:
    os.environ["SPS_HOME"] = "/opt/hostedtoolcache/Python/fsps"

import tables_io
from src.rail.creation.sed_generation import FSPSSedGenerator
import pytest
from src.rail.core.stage import RailStage
import numpy as np


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
        dummy_io_data = tables_io.read('src/rail/examples/testdata/test_fsps_sed.fits')
        sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', zcontinuous=1,
                                                          add_neb_emission=True, physical_units=True,
                                                          tabulated_sfh_file=None, tabulated_lsf_file=None,
                                                          **settings)
        sed_generation_test.add_data('input', dummy_io_data)
        sed_generation_test.run()


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
        dummy_io_data = tables_io.read('src/rail/examples/testdata/test_fsps_sed.fits')
        sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', zcontinuous=3,
                                                          add_neb_emission=True, physical_units=True,
                                                          tabulated_sfh_file=None, tabulated_lsf_file=None,
                                                          **settings)
        sed_generation_test.add_data('input', dummy_io_data)
        sed_generation_test.run()


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"sfh_type": 3}, FileNotFoundError),
    ],
)
def test_FSPSSedGenerator_missing_tabulated_sfh_file(settings, error):
    """Test bad tabulated sfh params that should raise Value and Type errors."""
    with pytest.raises(error):
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        dummy_io_data = tables_io.read('src/rail/examples/testdata/test_fsps_sed.fits')
        sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', zcontinuous=3,
                                                          add_neb_emission=False, physical_units=True,
                                                          tabulated_sfh_file='sfh.txt', tabulated_lsf_file=None,
                                                          **settings)
        sed_generation_test.add_data('input', dummy_io_data)
        sed_generation_test.run()


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"sfh_type": 3}, AssertionError),
    ],
)
def test_FSPSSedGenerator_wrong_age_tabulated_sfh_file(settings, error):
    """Test bad tabulated sfh params that should raise Value and Type errors."""
    with pytest.raises(error):
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        dummy_io_data = tables_io.read('src/rail/examples/testdata/test_fsps_sed.fits')
        sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', zcontinuous=3,
                                                          add_neb_emission=False, physical_units=True,
                                                          tabulated_sfh_file='src/rail/examples/testdata/'
                                                                             'sfh_bad_age_array.dat',
                                                          tabulated_lsf_file=None,
                                                          **settings)
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
        dummy_io_data = tables_io.read('src/rail/examples/testdata/test_fsps_sed.fits')
        sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', smooth_velocity=False,
                                                          physical_units=True, tabulated_sfh_file=None,
                                                          tabulated_lsf_file=None, **settings)
        sed_generation_test.add_data('input', dummy_io_data)
        sed_generation_test.run()


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"smooth_lsf": True}, FileNotFoundError),
    ],
)
def test_FSPSSedGenerator_missing_tabulated_lsf_file(settings, error):
    """Test bad tabulated sfh params that should raise Value and Type errors."""
    with pytest.raises(error):
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        dummy_io_data = tables_io.read('src/rail/examples/testdata/test_fsps_sed.fits')
        sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', smooth_velocity=True,
                                                          physical_units=True, tabulated_sfh_file=None,
                                                          tabulated_lsf_file='lsf.txt', **settings)
        sed_generation_test.add_data('input', dummy_io_data)
        sed_generation_test.run()


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"smooth_lsf": True}, ValueError),
    ],
)
def test_FSPSSedGenerator_wrong_wavelength_tabulated_lsf_file(settings, error):
    """Test bad tabulated sfh params that should raise Value and Type errors."""
    with pytest.raises(error):
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        dummy_io_data = tables_io.read('src/rail/examples/testdata/test_fsps_sed.fits')
        sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', smooth_velocity=True,
                                                          physical_units=True, tabulated_sfh_file=None,
                                                          tabulated_lsf_file='src/rail/examples/testdata/'
                                                                             'lsf_bad_wave_array.dat',
                                                          **settings)
        sed_generation_test.add_data('input', dummy_io_data)
        sed_generation_test.run()


def test_FSPSSedGenerator_output_table():
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    dummy_io_data = tables_io.read('src/rail/examples/testdata/test_fsps_sed.fits')
    sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', physical_units=True,
                                                      tabulated_sfh_file=None, tabulated_lsf_file=None)
    sed_generation_test.add_data('input', dummy_io_data)
    sed_generation_test.run()
    out_table = sed_generation_test.get_data('output')

    assert 'wavelength' in out_table.colnames
    assert 'spectrum' in out_table.colnames


def test_FSPSSedGenerator_non_physical_units():
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    dummy_io_data = tables_io.read('src/rail/examples/testdata/test_fsps_sed.fits')
    sed_generation_test = FSPSSedGenerator.make_stage(name='sed_generator_test', physical_units=False,
                                                      tabulated_sfh_file=None, tabulated_lsf_file=None)
    sed_generation_test.add_data('input', dummy_io_data)
    sed_generation_test.run()
    out_table = sed_generation_test.get_data('output')
    wave = out_table['wavelength']
    spec = out_table['spectrum']

    assert np.trapz(spec[0], x=wave[0]) < 1000
