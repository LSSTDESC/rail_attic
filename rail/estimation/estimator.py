import os
from rail.estimation.utils import *
import yaml
import importlib
import ast

class Estimator(object):
    """
    The base class for photo-z posterior estimates. inherit there will be a
    default loading of data (and write out of data?), but each code should have
    its own 'train' and 'estimate' methods that override the default methods
    in the parent class

    Super/subclass framework stolen shamelessly from
    https://github.com/LSSTDESC/tomo_challenge
    """

    base_dict = 'base.yaml'
    _subclasses = {}

    @classmethod
    def _find_subclass(cls, name):
        if name not in cls._subclasses.keys():
            root_dir = os.path.dirname(__file__)+'/algos'
            for filename in os.listdir(root_dir):
                if filename.endswith('.py') and filename != '__init__.py':
                    with open(os.path.join(root_dir,filename)) as f:
                        tree=ast.parse(f.read())
                        for node in tree.body:
                            if isinstance(node,ast.ClassDef) and node.name==name:
                                importlib.import_module('.'+filename[:-3],'rail.estimation.algos')
        
        return cls._subclasses[name]

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        print(f"Found classifier {cls.__name__}")
        cls._subclasses[cls.__name__] = cls

    def __init__(self, base_config='base_yaml', config_dict={}):
        if not os.path.exists(base_config):
            raise FileNotFoundError("File base_config="+base_config+" not found")

        with open(base_config, 'r') as f:
            base_dict = yaml.safe_load(f)['base_config']
        print('by request, base_dict='+str(base_dict))
        for n, v in base_dict.items():
            setattr(self, n, v)
        for attr in ['zmode', 'zgrid', 'pz_pdf']:
            setattr(self, attr, None)
        self.trainfile = base_dict['trainfile']
        self.outpath = base_dict['outpath']
        self.train_fmt = self.trainfile.split(".")[-1]

        self.groupname = base_dict['hdf5_groupname']
        self.training_data = load_training_data(self.trainfile, self.train_fmt,
                                                self.groupname)
        self.testfile = base_dict['testfile']
        self.num_rows = get_input_data_size_hdf5(self.testfile, self.groupname)
        self._chunk_size = base_dict['chunk_size']

        self.test_fmt = self.testfile.split(".")[-1]
        # self.test_data = load_data(self.testfile, self.test_fmt)
        # move reading of test data to main.py so we can loop more easily

        self.code_name = type(self).__name__

        self.config_dict = config_dict

    def inform(self):
        """
        Prior settings and/or training algorithm for the individual 
        photo-z method, should be implemented in the subclass
        """
        raise NotImplementedError

    def estimate(self, input_data):
        """
        The main run method for the photo-z, should be implemented in the specific
        subclass

        Input:
        ------
        data:
          dictionary of all input data

        Returns:
        --------
        pz_dict:
          dictionary of output photo-z params, must include zmode and pdf
          note: zgrid will still be a class variable for now

        should create photo-z estimates with set names, TBD
        for demo will just be `z_mode`
        """
        raise NotImplementedError
