import numpy as np
import os
from ..pipeutils import *

_PATH_TO_DATA = "/Users/sam/WORK/TOMOCHALLENGE"
_TRAIN_FILE = "mini_training.hdf5"
_TEST_FILE = "mini_validation.hdf5"

class BaseEstimation(object):
    """
    The base class from which specific methods should inherit there will be a 
    default loading of data (and write out of data?), but each code shoule have
    its own 'train' and 'run_photoz' methods that override the default methods in
    the parent class
    """
    
    def __init__(self, config_dict):

        self.basepath = config_dict.get('file_path', _PATH_TO_DATA)
        self.trainfile = config_dict.get('trainfile', _TRAIN_FILE)
        self.train_fmt = self.trainfile.split(".")[-1]
        self.training_data = load_data(self.trainfile,self.train_fmt)
        self.testfile = config_dict.get('testfile',_TEST_FILE)
        self.test_fmt = self.testfile.split(".")[-1]
        self.test_data = load_data(self.testfile,self.test_fmt)
        self.outfilebase = config_dict.get('outputfile', "generic_output.hdf5")
        self.code_name = config_dict.get('code_name', 'generic_code_name')

        fullpath = os.path.join(self.basepath,self.trainfile)


        
        

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
        fullname = self.code_name+"_"+self.outfilebase
        outf = h5py.File(fullname,"w")
        outf['photoz_mode'] = self.zmode
        outf['photoz_pdf']= self.pz_pdf
        outf['zgrid'] = self.zgrid
        outf.close()
