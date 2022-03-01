"""
Abstract base classes defining redshift estimations Trainers and Estimators
"""
import pickle

from rail.core.data import TableHandle, QPHandle, ModelHandle
from rail.core.types import DataFile
from rail.core.stage import RailStage


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
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=10000, hdf5_groupname=str)
    inputs = [('model_file', ModelHandle),
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
        self.finalize()
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
    config_options = RailStage.config_options.copy()
    config_options.update(hdf5_groupname=str, save_train=True)
    inputs = [('input', TableHandle)]
    outputs = [('model_file', ModelHandle)]

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
