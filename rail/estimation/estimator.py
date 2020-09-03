import numpy as np
import os
from utils import *

class Estimator(object):
    """
    The base class from which specific methods should inherit there will be a 
    default loading of data (and write out of data?), but each code should have
    its own 'train' and 'run_photoz' methods that override the default methods 
    in the parent class
    
    Super/subclass framework stolen shamelessly from 
    https://github.com/LSSTDESC/tomo_challenge
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
        self.num_rows = get_input_data_size_hdf5(self.testfile)
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
