"""
Abstract base classes defining redshift estimations Trainers and Estimators
"""
import pickle

from rail.core.data import TableHandle, QPHandle
from rail.core.types import DataFile
from rail.core.stage import RailStage


def default_model_read(modelfile):
    """Default function to read model files, simply used pickle.load"""
    return pickle.load(open(modelfile, 'rb'))

def default_model_write(modelfile, obj):
    """Default function to read model files, simply used pickle.load"""
    with open(modelfile) as f:
        pickle.dump(file=f, obj=obj, protocol=pickle.HIGHEST_PROTOCOL)



class ModelDict(dict):
    """
    A specialized dict to keep track of individual estimation models objects: this is just a dict these additional features

    1. Keys are paths
    2. There is a read(path, force=False) method that reads a model object and inserts it into the dictionary
    3. There is a write(path, model, force=False) method that write a model object and inserts it into the dictionary
    4. There is a single static instance of this class
    """
    def open(self, path, mode, **kwargs):  #pylint: disable=no-self-use
        """Open the file and return the file handle"""
        return open(path, mode, **kwargs)

    def read(self, path, force=False, reader=None):
        """Read a model into this dict"""
        if reader is None:
            reader = default_model_read
        if force or path not in self:
            model = reader(path)
            self.__setitem__(path, model)
            return model
        return self[path]

    def write(self, path, model, force=False, writer=None):
        """Read a model into this dict"""
        if writer is None:
            writer = default_model_write
        if force or path not in self:
            model = writer(path, model)
            self.__setitem__(path, model)
            return model
        return self[path]


MODEL_FACTORY = ModelDict()

def ModelFactory():
    """Return the singleton instance of the model factory"""
    return MODEL_FACTORY


class ModelFile(DataFile):
    """
    A file that describes an estimator mdoel
    """

    @classmethod
    def open(cls, path, mode, **kwargs):
        """ Opens a data file"""
        if mode == 'w':
            return MODEL_FACTORY.open(path, mode='wb', **kwargs)
        return MODEL_FACTORY.read(path, **kwargs)

    @classmethod
    def write(cls, data, path, **kwargs):
        """ Write a data file """
        return MODEL_FACTORY.write(path, data, **kwargs)


class Estimator(RailStage):
    """
    The base class for photo-z posterior estimates. inherit there will
    be a default loading of data (and write out of data?), but each code
    should have its own 'train' and 'estimate' methods that override the
    default methods in the parent class

    Super/subclass framework stolen shamelessly from
    https://github.com/LSSTDESC/tomo_challenge
    """

    name = 'Estimator'
    config_options = dict(chunk_size=10000, hdf5_groupname=str)
    inputs = [('model_file', ModelFile),
              ('input', TableHandle)]
    outputs = [('output', QPHandle)]

    def __init__(self, args, comm=None):
        """Initialize Estimator that can sample galaxy data."""
        RailStage.__init__(self, args, comm=comm)
        self.model = None
        self.open_model(**args)

    def open_model(self, **kwargs):
        """Load the model

        Keywords
        --------
        model : object
            An object with a trained model
        model_file : str
            A file from which to load a model object
        """
        model = kwargs.get('model', None)
        if model is not None:
            self.model = model
            self.config['model'] = None
            return
        model_file = kwargs.get('model_file', None)
        if model_file is not None:
            self.config['model_file'] = model_file
            if self.config['model_file'] is not None and self.config['model_file'] != 'None':
                self.model = self.open_input('model_file')

    def estimate(self, input_data):
        """
        The main run method for the photo-z, should be implemented
        in the specific subclass.

        Parameters
        ----------
        input_data : `dict`
          dictionary of all input data

        Returns
        -------
        output: `qp.Ensemble`
          Ensemble with output data
        """
        self.set_data('input', input_data)
        self.run()
        return self.get_handle('output')


class Trainer(RailStage):
    """
    The base class for photo-z posterior estimates. inherit there will
    be a default loading of data (and write out of data?), but each code
    should have its own 'train' and 'estimate' methods that override the
    default methods in the parent class

    Super/subclass framework stolen shamelessly from
    https://github.com/LSSTDESC/tomo_challenge
    """

    name = 'Trainer'
    config_options = dict(hdf5_groupname=str, save_train=True)
    inputs = [('input', TableHandle)]
    outputs = [('model_file', ModelFile)]

    def __init__(self, args, comm=None):
        """Initialize Trainer that can train models for redshift estimation """
        RailStage.__init__(self, args, comm=comm)
        self.model = None

    def write_model(self):
        """Write the model, this default implementation uses pickle"""
        with self.open_output('model_file') as f:
            pickle.dump(file=f, obj=self.model, protocol=pickle.HIGHEST_PROTOCOL)

    def inform(self, training_data):
        """
        The main run method for the photo-z, should be implemented
        in the specific subclass.

        Parameters
        ----------
        input_data : `dict`
          dictionary of all input data

        Returns
        -------
        output: `qp.Ensemble`
          Ensemble with output data
        """
        self.set_data('input', training_data)
        self.run()
