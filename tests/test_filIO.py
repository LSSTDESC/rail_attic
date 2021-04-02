import pytest
import qp
import os
import numpy as np
from rail.fileIO import load_training_data, initialize_qp_output
from rail.fileIO import write_qp_output_chunk, qp_reformat_output


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


def test_qp_file_writing():
    locs = np.array([1.0, 1.0, 2.0])
    locs2 = np.array([1.2, 1.9, 1.8])
    scales = np.array([0.1, 0.2, 0.5])
    ens = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))
    ens2 = qp.Ensemble(qp.stats.norm, data=dict(loc=locs2, scale=scales))
    outfile = "./fin_qp.hdf5"
    tmpfile = "./tmp_qp.hdf5"
    metafile = "./fin_qp_meta.hdf5"
    _ = initialize_qp_output(outfile)
    write_qp_output_chunk(tmpfile, outfile, ens, 0)
    write_qp_output_chunk(tmpfile, outfile, ens2, 1)
    num_chunks = 2
    qp_reformat_output(tmpfile, outfile, num_chunks)
    assert os.path.exists(outfile)
    assert os.path.exists(metafile)
    os.remove(outfile)
    os.remove(metafile)
