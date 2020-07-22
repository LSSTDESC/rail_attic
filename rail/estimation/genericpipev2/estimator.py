import numpy as np
import os
from .pipeutils import *

# #_PATH_TO_DATA = "/Users/sam/WORK/TOMOCHALLENGE"
# _PATH_TO_DATA = "/global/cfs/cdirs/lsst/groups/WL/users/zuntz/tomo_challenge_data/ugrizy"
# _TRAIN_FILE = "mini_training.hdf5"
# _TEST_FILE = "mini_validation.hdf5"

class Estimator(object):
    """
    The base class from which specific methods should inherit there will be a 
    default loading of data (and write out of data?), but each code should have
    its own 'train' and 'run_photoz' methods that override the default methods in
    the parent class
    
    Super/subclass framework stolen shamelessly from https://github.com/LSSTDESC/tomo_challenge
    """
    
#     base_dict = 'base.yaml'
    
    _subclasses = {}

    @classmethod
    def _find_subclass(cls, name):
        return cls._subclasses[name]

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        print(f"Found classifier {cls.__name__}")
        cls._subclasses[cls.__name__] = cls
        config_yaml = 'genericpipev2/examples/' + cls.__name__ + '.yaml'#config_dict['base_yaml']
#         with open(config_yaml, 'r') as f:
#             cls.config_dict = yaml.safe_load(f)
#         print('by request, config_dict='+str(cls.config_dict))
    
    def __init__(self, config_dict):
#         code_dict = base_dict['run_params']
#         base_dict = base_dict['base_config']
        base_yaml = 'genericpipev2/base.yaml'#config_dict['base_yaml']
        with open(base_yaml, 'r') as f:
            base_dict = yaml.safe_load(f)['base_config']
        print('by request, base_dict='+str(base_dict))
        
#         self.basepath = config_dict.get('file_path', _PATH_TO_DATA)
        self.trainfile = base_dict['trainfile']#(.get(, _TRAIN_FILE)
        self.train_fmt = self.trainfile.split(".")[-1]
        self.training_data = load_data(self.trainfile, self.train_fmt)
        self.testfile = base_dict['testfile']#.get(,_TEST_FILE)
        self.test_fmt = self.testfile.split(".")[-1]
        self.test_data = load_data(self.testfile, self.test_fmt)
#         fullpath = os.path.join(self.basepath,self.trainfile)
        
        self.saveloc = os.path.join(base_dict['outpath'], type(self).__name__ + '.hdf5')#config_dict.get('outputfile', "generic_output.hdf5")

#                 self.outfilebase = code_dict.get('outputfile', "generic_output.hdf5")
        self.code_name = type(self).__name__#code_dict.get('code_name', 'generic_code_name')
    
#         with open(config_yaml, 'r') as f:
#             self.config_dict = yaml.safe_load(f)
#         print('by request, base_dict='+str(base_dict))
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
        fullname = os.path.join('', self.code_name+'.hdf5')#+"_"+self.outfilebase
        outf = h5py.File(fullname,"w")
        outf['photoz_mode'] = self.zmode
        outf['photoz_pdf'] = self.pz_pdf
        outf['zgrid'] = self.zgrid
        outf.close()
