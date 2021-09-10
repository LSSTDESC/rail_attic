import pytest
import qp
import os
import numpy as np
from rail.fileIO import initialize_qp_output
from rail.fileIO import write_qp_output_chunk, qp_reformat_output


h5_data_file = 'tests/data/pandas_test_hdf5.h5'
parquet_data_file = 'tests/data/parquet_test.parquet'
no_group_file = 'tests/data/no_groupname_test.hdf5'


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


def test_qp_existing_file_overwrite():
    testfile = "./existing_file.hdf5"
    metafile = "./existing_file_meta.hdf5"
    open(testfile, "a").close()  # create fake empty files
    open(metafile, "a").close()
    initialize_qp_output(testfile)
    assert not os.path.exists(testfile)  # check that files
    assert not os.path.exists(metafile)  # deleted


def test_subdir_creation():
    testfile = "./FAKEDIR/test.hdf5"
    initialize_writeout(testfile, 5, 5)
    assert os.path.exists(testfile)
    os.remove(testfile)
    os.rmdir("FAKEDIR")
    # test dir creation of qp func as well
    initialize_qp_output(testfile)
    assert (os.path.exists("./FAKEDIR"))
    os.rmdir("FAKEDIR")
