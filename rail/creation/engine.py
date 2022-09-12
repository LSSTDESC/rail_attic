"""
Abstract base classes defining a Creator, which will create synthetic photometric data
and a PosteriorCalculator, which can calculate posteriors for the data with respect
to the distribution defined by the creator.
"""

import pandas as pd
import qp
from rail.core.data import DataHandle, ModelHandle, QPHandle, TableHandle
from rail.core.stage import RailStage


class Modeler(RailStage):
    """
    Base class for creating a model of redshift and photometry.

    """

    name = "Modeler"
    config_options = RailStage.config_options.copy()
    config_options.update(seed=12345)
    inputs = [("input", DataHandle)]
    outputs = [("model", ModelHandle)]

    def __init__(self, args, comm=None):
        """Initialize Modeler"""
        RailStage.__init__(self, args, comm=comm)
        self.model = None

    def fit_model(self):
        """
        Produce a creation model from which photometry and redshifts can be generated

        Parameters
        ----------
        [The parameters depend entirely on the modeling approach!]

        Returns
        -------
        [This will definitely be a file, but the filetype and format depend entirely on the modeling approach!]
        """
        self.run()
        self.finalize()
        return self.get_handle("model")


class Creator(RailStage):
    """Base class for Creators that generate synthetic photometric data from a model.

    `Creator` will output a table of photometric data.  The details
    will depend on the particular engine.
    """

    name = "Creator"
    config_options = RailStage.config_options.copy()
    config_options.update(n_samples=int, seed=12345)
    inputs = [("model", ModelHandle)]
    outputs = [("output", TableHandle)]

    def __init__(self, args, comm=None):
        """Initialize Creator"""
        RailStage.__init__(self, args, comm=comm)
        self.model = None
        if not isinstance(args, dict):  # pragma: no cover
            args = vars(args)
        self.open_model(**args)

    def open_model(self, **kwargs):
        """Load the mode and/or attach it to this Creator

        Keywords
        --------
        model : `object`, `str` or `ModelHandle`
            Either an object with a trained model,
            a path pointing to a file that can be read to obtain the trained model,
            or a `ModelHandle` providing access to the trained model.

        Returns
        -------
        self.model : `object`
            The object encapsulating the trained model.
        """
        model = kwargs.get("model", None)
        if model is None or model == "None":  # pragma: no cover
            self.model = None
            return self.model
        if isinstance(model, str):  # pragma: no cover
            self.model = self.set_data("model", data=None, path=model)
            self.config["model"] = model
            return self.model
        if isinstance(model, ModelHandle):  # pragma: no cover
            if model.has_path:
                self.config["model"] = model.path
        self.model = self.set_data("model", model)
        return self.model

    def sample(self, n_samples: int, seed: int = None, **kwargs):
        """Draw samples from the model specified in the configuration.

        This is a method for running a Creator in interactive mode.
        In pipeline mode, the subclass `run` method will be called by itself.

        Parameters
        ----------
        n_samples: int
            The number of samples to draw
        seed: int
            The random seed to control sampling

        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame of the samples

        Notes
        -----
        This method puts `n_samples` and `seed` into the stage configuration
        data, which makes them available to other methods.
        It then calls the `run` method, which must be defined by a subclass.
        Finally, the `DataHandle` associated to the `output` tag is returned.
        """
        self.config["n_samples"] = n_samples
        self.config["seed"] = seed
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        return self.get_handle("output")


class PosteriorCalculator(RailStage):
    """Base class for object that calculates the posterior distribution of a
    particular field in a table of photometric data  (typically the redshift).

    The posteriors will be contained in a qp Ensemble.
    """

    name = "PosteriorCalculator"
    config_options = RailStage.config_options.copy()
    config_options.update(column=str)
    inputs = [
        ("model", ModelHandle),
        ("input", TableHandle),
    ]
    outputs = [("output", QPHandle)]

    def __init__(self, args, comm=None):
        """Initialize PosteriorCalculator"""
        RailStage.__init__(self, args, comm=comm)
        self.model = None
        if not isinstance(args, dict):  # pragma: no cover
            args = vars(args)
        self.open_model(**args)

    def open_model(self, **kwargs):
        """Load the mode and/or attach it to this PosteriorCalculator

        Keywords
        --------
        model : `object`, `str` or `ModelHandle`
            Either an object with a trained model,
            a path pointing to a file that can be read to obtain the trained model,
            or a `ModelHandle` providing access to the trained model.

        Returns
        -------
        self.model : `object`
            The object encapsulating the trained model.
        """
        model = kwargs.get("model", None)
        if model is None or model == "None":  # pragma: no cover
            self.model = None
            return self.model
        if isinstance(model, str):  # pragma: no cover
            self.model = self.set_data("model", data=None, path=model)
            self.config["model"] = model
            return self.model
        if isinstance(model, ModelHandle):  # pragma: no cover
            if model.has_path:
                self.config["model"] = model.path
        self.model = self.set_data("model", model)
        return self.model

    def get_posterior(self, input_data: pd.DataFrame, **kwargs) -> qp.Ensemble:
        """Return posteriors for the given column.

        This is a method for running a Creator in interactive mode.
        In pipeline mode, the subclass `run` method will be called by itself.

        Parameters
        ----------
        data: pd.DataFrame
            A Pandas DataFrame of the galaxies for which posteriors are calculated

        Notes
        -----
        This will put the `data` argument input this Stages the DataStore using this stages `input` tag.
        This will put the additional functional arguments into this Stages configuration data.

        It will then call `self.run()` and return the `DataHandle` associated to the `output` tag
        """
        self.set_data("input", input_data)
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        return self.get_handle("output")
