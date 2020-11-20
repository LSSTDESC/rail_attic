import pytest
from rail.estimation.utils import load_training_data


h5_data_file = 'tests/data/pandas_test_hdf5.h5'
parquet_data_file = 'tests/data/parquet_test.parquet'
no_group_file = 'tests/data/no_groupname_test.hdf5'


def test_pandas_readin():
    _ = load_training_data(h5_data_file, fmt='h5')
    _ = load_training_data(parquet_data_file, fmt='parquet')


def test_no_groupname():
    _ = load_training_data(no_group_file, fmt='hdf5', groupname='None')


def test_missing_file_ext():
    with pytest.raises(NotImplementedError):
        _ = load_training_data(no_group_file, fmt='csv')
