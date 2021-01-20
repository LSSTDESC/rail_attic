# Note: 'utils.py' is terrible!  These i/o functions don't really belong
# here, so the bad name is a reminder to deal with this.
import os
import h5py
import pandas as pd
import numpy as np


def load_training_data(filename, fmt='hdf5', groupname='None'):
    fmtlist = ['hdf5', 'parquet', 'h5']
    if fmt not in fmtlist:
        raise NotImplementedError(f"File format {fmt} not implemented")
    if fmt == 'hdf5':
        data = load_raw_hdf5_data(filename, groupname)
    if fmt == 'parquet':
        data = load_raw_pq_data(filename)
    if fmt == 'h5':
        data = load_raw_h5_data(filename)
    return data


def load_raw_pq_data(infile):
    """
    just return the dataframe from pandas for now
    """
    df = pd.read_parquet(infile, engine='pyarrow')
    data = {}
    for key in df.keys():
        data[key] = np.array(df[key])
    return data


def load_raw_h5_data(infile):
    """just return the datafram from pandas h5"""
    df = pd.read_hdf(infile)
    data = {}
    for key in df.keys():
        data[key] = np.array(df[key])
    return data


def load_raw_hdf5_data(infile, groupname='None'):
    """
    read in h5py hdf5 data, return a dictionary of all of the keys
    """
    data = {}
    infp = h5py.File(infile, "r")
    if groupname != 'None':
        f = infp[groupname]
    else:
        f = infp
    for key in f.keys():
        data[key] = np.array(f[key])
    infp.close()
    return data


def get_input_data_size_hdf5(infile, groupname='None'):
    infp = h5py.File(infile, "r")
    if groupname != 'None':
        f = infp[groupname]
    else:
        f = infp
    firstkey = list(f.keys())[0]
    nrows = len(f[firstkey])
    infp.close()
    return nrows


def iter_chunk_hdf5_data(infile, chunk_size=100_000, groupname='None'):
    """
    itrator for sending chunks of data in hdf5.
    input:
    ------
      infile: input file name (str)
      chunk_size: size of chunk to iterate over (int)
    output: interator chunk consisting of dictionary of all the keys
    Currently only implemented for hdf5
      start: start index (int)
      end: ending index (int)
      data: dictionary of all data from start:end (dict)
    """
    data = {}
    num_rows = get_input_data_size_hdf5(infile, groupname)
    infp = h5py.File(infile, "r")
    if groupname != 'None':
        f = infp[groupname]
    else:
        f = infp
    for i in range(0, num_rows, chunk_size):
        start = i
        end = i+chunk_size
        if end > num_rows:
            end = num_rows
        for key in f.keys():
            data[key] = np.array(f[key][start:end])
        yield start, end, data
    infp.close()


def initialize_writeout(outfile, num_rows, num_zbins):
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    outf = h5py.File(outfile, "w")
    outf.create_dataset('photoz_mode', (num_rows,), dtype='f4')
    outf.create_dataset('photoz_pdf', (num_rows, num_zbins), dtype='f4')
    return outf


def write_out_chunk(outf, data_dict, start, end):
    outf['photoz_mode'][start:end] = data_dict['zmode']
    outf['photoz_pdf'][start:end] = data_dict['pz_pdf']


def finalize_writeout(outf, zgrid):
    outf['zgrid'] = zgrid
    outf.close()


def write_output_file(outfile, num_rows, num_zbins, data_dict, zgrid):
    outf = initialize_writeout(outfile, num_rows, num_zbins)
    write_out_chunk(outf, data_dict, 0, num_zbins)
    finalize_writeout(outf, zgrid)
