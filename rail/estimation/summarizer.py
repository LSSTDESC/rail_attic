"""
Abstract base classes defining redshift estimations Informers and Estimators
"""
from rail.core.data import QPHandle, TableHandle, ModelHandle
from rail.core.stage import RailStage



class CatSummarizer(RailStage):  #pragma: no cover
    """The base class for classes that go from catalog-like tables
    to ensemble NZ estimates.

    CatSummarizer take as "input" a catalog-like table.  I.e., a
    table with fluxes in photometric bands among the set of columns.

    provide as "output" a QPEnsemble, with per-ensemble n(z).
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
        """The main run method for the summarization, should be implemented
        in the specific subclass.

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





class PzInformer(RailStage):  #pragma: no cover
    """The base class for informing models used to summarize photo-z posterior estimates
    from ensembles of p(z) distributions.

    PzSummarizers can use a generic "model", the details of which depends on the sub-class.
    Some summaer will have associated PzInformer classes, which can be used to inform
    those models.

    (Note, "Inform" is more generic than "Train" as it also applies to algorithms that
    are template-based rather than machine learning-based.)

    PzInformer will produce as output a generic "model", the details of which depends on the sub-class.

    They take as "input" a qp.Ensemble of per-galaxy p(z) data, which is used to "inform" the model.
    """

    name = 'Informer'
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
