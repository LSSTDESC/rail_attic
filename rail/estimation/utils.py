import os
import yaml
import h5py
import pandas as pd
import numpy as np


#PATH_TO_DATA = "/Users/sam/WORK/TOMOCHALLENGE"
# PATH_TO_DATA = "/global/cfs/cdirs/lsst/groups/WL/users/zuntz/tomo_challenge_data/ugrizy"
bands = ['u','g','r','i','z','y']

base_yaml = 'base.yaml'#config_dict['base_yaml']
with open(base_yaml, 'r') as f:
    base_dict = yaml.safe_load(f)['base_config']

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

def load_raw_h5_data(infile):
    """just return the dataframe from pandas h5"""
    return ps.read_hdf(filename)

def load_raw_pq_data(infile):
    """ 
    just return the dataframe from pandas for now
    """
    return pd.read_parquet(filename,engine='pyarrow')
    

def load_data(filename, fmt='hdf5'):
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

    
def transform_data(input_data):
    """
      Maybe want to add colors, color errors for some codes to use
      Or, just leave it up to the individual codes?
    """
    pass


