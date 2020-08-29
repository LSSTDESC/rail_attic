import numpy as np
import os
from utils import *

class Estimator(object):
    """
    The base class from which specific methods should inherit there will be a 
    default loading of data (and write out of data?), but each code should have
    its own 'train' and 'run_photoz' methods that override the default methods in
    the parent class
    
    Super/subclass framework stolen shamelessly from https://github.com/LSSTDESC/tomo_challenge
    """
    
    base_dict = 'base.yaml'
    
    _subclasses = {}

    @classmethod
    def _find_subclass(cls, name):
        return cls._subclasses[name]

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        print(f"Found classifier {cls.__name__}")
        cls._subclasses[cls.__name__] = cls
        config_yaml = '.configs/' + cls.__name__ + '.yaml'
    
    def __init__(self, config_dict):

        with open(base_yaml, 'r') as f:
            base_dict = yaml.safe_load(f)['base_config']
        print('by request, base_dict='+str(base_dict))
        
        self.trainfile = base_dict['trainfile']
        self.train_fmt = self.trainfile.split(".")[-1]
        self.training_data = load_training_data(self.trainfile, self.train_fmt)
        self.testfile = base_dict['testfile']
        self._chunk_size = base_dict['chunk_size']
        self.test_fmt = self.testfile.split(".")[-1]
        # self.test_data = load_data(self.testfile, self.test_fmt)
        # move reading of test data to main.py so we can loop more easily
        
        self.code_name = type(self).__name__
        self.saveloc = os.path.join(base_dict['outpath'], self.code_name + '.hdf5')
        
        self.config_dict = config_dict

    def train(self):
        """
        A training algorithm for the individual photo-z method, should be
        implemented in the subclass
        """
        pass

    def run_photoz(self):
        """
        The main run method for the photo-z, should be implemented in the specific
        subclass

        should create photo-z estimates with set names, TBD
        for demo will just be `z_mode`
        """
        pass

    def initialize_writeout(self):
        outf = h5py.File(self.saveloc,"w")
        outf.create_dataset('photoz_mode', (self.num_rows,), dtype='f4')
        outf.create_dataset('photoz_pdf', (self.num_rows,self.nzbins),
                            dtype='f4')
        return outf
        
    def write_out_chunk(self, outf, start, end):
        outf['photoz_mode'][start:end] = self.zmode
        outf['photoz_pdf'][start:end] = self.pz_pdf
        
    def finalize_writeout(self,outf):
        outf['zgrid'] = self.zgrid
        outf.close()

    def load_data(self, filename, fmt='hdf5'):
        fmtlist = ['hdf5', 'parquet', 'h5']
        if fmt not in fmtlist:
            raise ValueError(f"File format {fmt} not implemented")
        if fmt == 'hdf5':
            data = iter_chunk_hdf5_data(filename)
        if fmt == 'parquet':
            raise ValueError("parllel loading of parquet not yet implemented")
            # data = load_raw_pq_data(filename)
        if fmt == 'h5':
            raise ValueError("parallel loading of pandas h5 not implemented")
            # data = load_raw_h5_data(filename)
        return data

    def iter_chunk_hdf5_data(self,infile):
        """                                        
        itrator for sending chunks of data in hdf5.
        input: input filename                                           
        output: interator chunk consisting of dictionary of all the keys
        Currently only implemented for hdf5
        """
        data = {}
        f = h5py.File(infile,"r")
        firstkey = list(f.keys())[0]
        self.num_rows = len(f[firstkey])
        for i in range(0,self.num_rows,self._chunk_size):
            start = i
            end = i+self._chunk_size
            if end > self.num_rows:
                end = self.num_rows
            for key in f.keys():
                data[key] = np.array(f[key][start:end])
            yield start, end, data

        
