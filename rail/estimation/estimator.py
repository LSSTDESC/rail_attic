"""
Abstract base classes defining redshift estimations Informers and Estimators
"""

from rail.core.data import TableHandle, QPHandle, ModelHandle
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
    config_options.update(chunk_size=10000, hdf5_groupname=str,
                         mag_colname='mag_{}_lsst',
                         mag_err_colname='mag_err_{}_lsst',
                         redshift_colname='redshift')
    inputs = [('model', ModelHandle),
              ('input', TableHandle)]
    outputs = [('output', QPHandle)]

    def __init__(self, args, comm=None):
        """Initialize Estimator that can sample galaxy data."""
        RailStage.__init__(self, args, comm=comm)
        self.model = None
        if not isinstance(args, dict):  #pragma: no cover
            args = vars(args)
        self.open_model(**args)
        self._make_mag_colname()

    def _make_mag_colname(self):
        
        # This function specifies the column names of magnitudes and their errors
        self.mag_cols = {b: self.config['mag_colname'].replace('{}', b) 
                         for b in self.config.bands}
        self.mag_err_cols = {b: self.config['mag_err_colname'].replace('{}', b) 
                             for b in self.config.bands}
        
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
        if model is None or model == 'None':
            self.model = None
            return self.model
        if isinstance(model, str):
            self.model = self.set_data('model', data=None, path=model)
            self.config['model'] = model
            return self.model
        if isinstance(model, ModelHandle):
            if model.has_path:
                self.config['model'] = model.path
        self.model = self.set_data('model', model)
        return self.model

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


class Informer(RailStage):
    """
    The base class for photo-z posterior estimates. inherit there will
    be a default loading of data (and write out of data?), but each code
    should have its own 'train' and 'estimate' methods that override the
    default methods in the parent class

    Super/subclass framework stolen shamelessly from
    https://github.com/LSSTDESC/tomo_challenge
    """

    name = 'Informer'
    config_options = RailStage.config_options.copy()
    config_options.update(hdf5_groupname=str, save_train=True)
    inputs = [('input', TableHandle)]
    outputs = [('model', ModelHandle)]

    def __init__(self, args, comm=None):
        """Initialize Informer that can train models for redshift estimation """
        RailStage.__init__(self, args, comm=comm)
        self.model = None

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
        self.finalize()
