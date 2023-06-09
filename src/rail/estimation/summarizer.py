"""
Abstract base classes for characterizing redshift distributions for an ensemble of galaxies.
"""
from rail.core.data import QPHandle, TableHandle, ModelHandle
from rail.core.stage import RailStage

class PzInformer(RailStage):  #pragma: no cover
    """The base class for informing models used to summarize photo-z posterior estimates
    from ensembles of p(z) distributions.

    Summarizers take as input a generic "model", the details of which depend on the sub-class.
    All Summarizes must have an associated Informer that produces the models given inputs such as training sets or SED template libraries with priors, hence the generic name; while "Trainer" would be accurate for data-driven estimators, "Informer" also encompases model-fitting methods ingesting prior information.

    They take as "input" a qp.Ensemble of per-galaxy p(z) data, which is used to "inform" the model.
    """

    name = 'PzInformer'
    config_options = RailStage.config_options.copy()
    inputs = [('input', QPHandle)]
    outputs = [('model', ModelHandle)]

    def __init__(self, args, comm=None):
        """Initialize Informer that can inform models for redshift estimation """
        RailStage.__init__(self, args, comm=comm)
        self.model = None

    def inform(self, training_data):
        """The main interface method for Informers

        This will attach the input_data to this `Informer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the model that it creates to this Estimator
        by using `self.add_data('model', model)`.

        Finally, this will return a ModelHandle providing access to the trained model.

        Parameters
        ----------
        input_data :  `qp.Ensemble`
            Per-galaxy p(z), and any ancilary data associated with it

        Returns
        -------
        model : ModelHandle
            Handle providing access to trained model
        """
        self.set_data('input', training_data)
        self.run()
        self.finalize()
        return self.get_handle('model')



class CatSummarizer(RailStage):  #pragma: no cover
    """The base class to go from catalog-like tables
    to ensemble N(z) estimates.

    CatSummarizer takes as "input" a catalog-like table, i.e., a
    table with fluxes in photometric bands among the set of columns.

    It provides as "output" a QPEnsemble, with a single n(z) or samples thereof.
    """

    name = 'CatSummarizer'
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=10000)
    inputs = [('input', TableHandle)]
    outputs = [('output', QPHandle)]

    def __init__(self, args, comm=None):
        """Initialize Summarizer"""
        RailStage.__init__(self, args, comm=comm)

    def summarize(self, input_data):
        """The main method for the summarization process implemented
        in the run method of each specific subclass.

        This will attach the input_data to this `CatSummarizer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this `CatSummarizer`
        by using `self.add_data('output', output_data)`.

        Finally, this will return a QPHandle providing access to that output data.

        Parameters
        ----------
        input_data : `dict` or `ModelHandle`
            Either a dictionary of all input data or a `ModelHandle` providing access to the same

        Returns
        -------
        output: `qp.Ensemble`
            Ensemble with n(z), and any ancilary data
        """
        self.set_data('input', input_data)
        self.run()
        self.finalize()
        return self.get_handle('output')


class PZSummarizer(RailStage):
    """The base class for classes that go from per-galaxy PZ estimates to ensemble NZ estimates

    PZSummarizer take as "input" a `qp.Ensemble` with per-galaxy PDFs, and
    provide as "output" a QPEnsemble, with per-ensemble n(z).
    """

    name = 'PZtoNZSummarizer'
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=10000)
    inputs = [('input', QPHandle)]
    outputs = [('output', QPHandle)]

    def __init__(self, args, comm=None):
        """Initialize Estimator that can sample galaxy data."""
        RailStage.__init__(self, args, comm=comm)

    def summarize(self, input_data):
        """The main run method for the summarization, should be implemented
        in the specific subclass.

        This will attach the input_data to this `PZtoNZSummarizer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Estimator
        by using `self.add_data('output', output_data)`.

        Finally, this will return a QPHandle providing access to that output data.

        Parameters
        ----------
        input_data : `qp.Ensemble`
            Per-galaxy p(z), and any ancilary data associated with it

        Returns
        -------
        output: `qp.Ensemble`
            Ensemble with n(z), and any ancilary data
        """
        self.set_data('input', input_data)
        self.run()
        self.finalize()
        return self.get_handle('output')


class SZPZSummarizer(RailStage):
    """The base class for classes that use two sets of data: a photometry sample with
    spec-z values, and a photometry sample with unknown redshifts, e.g. simpleSOM and
    outputs a QP Ensemble with bootstrap realization of the N(z) distribution
    """
    name = 'SZPZtoNZSummarizer'
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=10000)
    inputs = [('input', TableHandle),
              ('spec_input', TableHandle),
              ('model', ModelHandle)]
    outputs = [('output', QPHandle)]

    def __init__(self, args, comm=None):
        """Initialize Estimator that can sample galaxy data."""
        RailStage.__init__(self, args, comm=comm)
        self.model = None
        if not isinstance(args, dict):  #pragma: no cover
            args = vars(args)
        self.open_model(**args)

    def open_model(self, **kwargs):
        """Load the mode and/or attach it to this Summarizer

        Notes
        -----
        Keywords are potentially used to build the model.
        By default we just use 'model'  which can be an `object`, `str` or `ModelHandle`

        I.e., either an object with a trained model, a path pointing to a file that
        can be read to obtain the trained model, or a `ModelHandle` providing access to the trained model.

        Returns
        -------
        self.model : `object`
            The object encapsulating the trained model.
        """
        model = kwargs.get('model', None)
        if model is None or model == 'None':  # pragma: no cover
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

    def summarize(self, input_data, spec_data):
        """The main run method for the summarization, should be implemented
        in the specific subclass.

        This will attach the input_data to this `SZandPhottoNZSummarizer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Estimator
        by using `self.add_data('output', output_data)`.

        Finally, this will return a QPHandle providing access to that output data.

        Parameters
        ----------
        input_data : `qp.Ensemble`
            Per-galaxy p(z), and any ancilary data associated with it

        Returns
        -------
        output: `qp.Ensemble`
            Ensemble with n(z), and any ancilary data
        """
        self.set_data('input', input_data)
        self.set_data('spec_input', spec_data)
        self.run()
        self.finalize()
        return self.get_handle('output')
