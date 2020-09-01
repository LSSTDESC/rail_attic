#Note: 'utils.py' is terrible!  These i/o functions don't really belong here, so the bad name is a reminder to deal with this.
import os
import yaml
import h5py
import pandas as pd
import numpy as np

#Note: 'base.yaml' shouldn't be hardcoded here!  It belongs in 'main.py'.
base_yaml = 'base.yaml'
with open(base_yaml, 'r') as f:
    base_dict = yaml.safe_load(f)['base_config']

def load_training_data(filename, fmt='hdf5'):
    fmtlist = ['hdf5', 'parquet', 'h5']
    if fmt not in fmtlist:
        raise ValueError(f"File format {fmt} not implemented")
    if fmt == 'hdf5':
        data = load_raw_hdf5_data(filename)
    if fmt == 'parquet':
        data = load_raw_pq_data(filename)                                      
    if fmt == 'h5':
        data = load_raw_h5_data(filename)                                     
    return data

def load_raw_pq_data(infile):
    """                       
    just return the dataframe from pandas for now
    """
    return pd.read_parquet(filename,engine='pyarrow')

def load_raw_h5_data(infile):
    """just return the datafram from pandas h5"""
    return pd.read_hdf(filename)

def load_raw_hdf5_data(infile):
    """                                                                         
    read in h5py hdf5 data, return a dictionary of all of the keys              
    """
    data = {}
    f = h5py.File(infile, "r")
    for key in f.keys():
        data[key] = np.array(f[key])
    f.close()
    return data

def get_input_data_size_hdf5(infile):
    f = h5py.File(infile,"r")
    firstkey = list(f.keys())[0]
    return len(f[firstkey])

def iter_chunk_hdf5_data(infile,chunk_size=100_000):
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
    num_rows = get_input_data_size_hdf5(infile)
    f = h5py.File(infile,"r")
    for i in range(0,num_rows,chunk_size):
        start = i
        end = i+chunk_size
        if end > num_rows:
            end = num_rows
        for key in f.keys():
            data[key] = np.array(f[key][start:end])
        yield start, end, data
    f.close() #does this work?

