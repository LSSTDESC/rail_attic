"""
Abstract base classes defining redshift estimations Trainers and Estimators
"""
from rail.core.data import QPHandle, TableHandle
from rail.core.stage import RailStage


class PZtoNZSummarizer(RailStage):
    """
    The base class for classes that go from per-galaxy PZ estimates to ensemble NZ estimates
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
        """
        The main run method for the summarization, should be implemented
        in the specific subclass.

        Parameters
        ----------
        input_data : `qp.Ensemble`
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
