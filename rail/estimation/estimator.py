import numpy as np
import os
from rail.estimation.utils import *


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
        config_yaml = '.configs/' + cls.__name__ + '.yaml'#config_dict['base_yaml']
#         with open(config_yaml, 'r') as f:
#             cls.config_dict = yaml.safe_load(f)
#         print('by request, config_dict='+str(cls.config_dict))
    
    def __init__(self, base_config='base_yaml', config_dict={}):
        if not os.path.exists(base_config):
            raise FileNotFoundError("File base_config="+base_config+" not found")

        with open(base_config, 'r') as f:
            base_dict = yaml.safe_load(f)['base_config']
        print('by request, base_dict='+str(base_dict))
        for n,v in base_dict.items():
            setattr(self, n, v)
        for attr in ['zmode','zgrid','pz_pdf']:
            setattr(self,attr,None)
        
        self.train_fmt = self.trainfile.split(".")[-1]
        self.training_data = load_data(self.trainfile, self.train_fmt)
        self.test_fmt = self.testfile.split(".")[-1]
        self.test_data = load_data(self.testfile, self.test_fmt)
        
        self.code_name = type(self).__name__
        self.saveloc = os.path.join(self.outpath, self.code_name + '.hdf5')
    
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

    def write_out(self):
        print("write out function")
        fullname = self.saveloc
        outf = h5py.File(fullname,"w")
        outf['photoz_mode'] = self.zmode
        outf['photoz_pdf'] = self.pz_pdf
        outf['zgrid'] = self.zgrid
        outf.close()
