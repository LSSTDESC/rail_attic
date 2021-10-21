import os
from tables_io.ioUtils import getInputDataLengthHdf5
import yaml
import pickle
import pprint


class Estimator(object):
    """
    The base class for photo-z posterior estimates. inherit there will
    be a default loading of data (and write out of data?), but each code
    should have its own 'train' and 'estimate' methods that override the
    default methods in the parent class

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

    def __init__(self, base_config='base_yaml', config_dict={}):
        # Allow estimators to be configured either with a dict
        # that has already been ready or with the yaml file directly
        if isinstance(base_config, dict):
            base_dict = base_config
        else:
            if not os.path.exists(base_config):
                raise FileNotFoundError("File base_config=" + base_config
                                        + " not found")

            with open(base_config, 'r') as f:
                base_dict = yaml.safe_load(f)['base_config']

        # Pretty-print the configuration
        print('Basic estimator configuration: ')
        pprint.pprint(base_dict)

        for n, v in base_dict.items():
            setattr(self, n, v)
        for attr in ['zmode', 'zgrid', 'pz_pdf']:
            setattr(self, attr, None)
        self.outpath = base_dict['outpath']

        self.trainfile = base_dict['trainfile']
        self.groupname = base_dict['hdf5_groupname']
        self.testfile = base_dict['testfile']
        self.num_rows = getInputDataLengthHdf5(self.testfile, self.groupname)
        self._chunk_size = base_dict['chunk_size']

        self.output_format = base_dict['output_format']

        self.test_fmt = self.testfile.split(".")[-1]
        # self.test_data = load_data(self.testfile, self.test_fmt)
        # move reading of test data to main.py so we can loop more easily

        self.code_name = type(self).__name__

        self.config_dict = config_dict

    def inform(self, training_data):
        """
        Prior settings and/or training algorithm for the individual
        photo-z method, should be implemented in the subclass
        Input:
        ------
        training_data: dict
          dictionary of the training data, *including* redshift
        """
        raise NotImplementedError

    def load_pretrained_model(self):
        """
        If inform step has been run separately, this funciton will
        load the information required to run estimate.  As a
        default we will include the loading of a pickled model,
        but the idea is that a specific code can override this
        function by writing a custom model load in the subclass
        """
        try:
            modelfile = self.inform_options['modelfile']
        except KeyError:
            print("inform_options['modelfile'] not specified, exiting!")
            raise KeyError("inform_options['modelfile'] not found!")
        try:
            self.model = pickle.load(open(modelfile, 'rb'))
            print(f"success in loading {self.inform_options['modelfile']}")
        except FileNotFoundError:
            print(f"File {self.inform_options['modelfile']} not found!")
            raise FileNotFoundError("File " +
                                    self.inform_options['modelfile'] +
                                    " not found!")

    def estimate(self, input_data):
        """
        The main run method for the photo-z, should be implemented 
        in the specific subclass.

        Parameters
        ----------
        data : `dict`
          dictionary of all input data

        Returns
        -------
        pz_dict : `dict`
          dictionary of output photo-z params, must include zmode and
          pdf 
        """
        # note: zgrid will still be a class variable for now
        # should create photo-z estimates with set names, TBD
        # for demo will just be `z_mode`
        raise NotImplementedError
