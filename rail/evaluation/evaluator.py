

class Evaluator:
    """ A superclass for metrics evaluations"""
    def __init__(self, qp_ens):
        """Class constructor.
        Parameters
        ----------
        qp_ens: qp.Ensemble object
            PDFs as qp.Ensemble
        """
        self._qp_ens = qp_ens

    def evaluate(self):  #pragma: no cover
        """
        Evaluates the metric a function of the truth and prediction

        Returns
        -------
        metric: dictionary
            value of the metric and statistics thereof
        """
        raise NotImplementedError


# class CRPS(Evaluator):
#     ''' Continuous rank probability score (Gneiting et al., 2006)'''
#
#     def __init__(self, sample, name="CRPS"):
#         """Class constructor.
#         Parameters
#         ----------
#         sample: `qp.ensemble`
#             ensemble of PDFS
#         name: `str`
#             the name of the metric
#         """
#         super().__init__(sample, name)
#
#
#     def evaluate(self):
#         raise NotImplementedError
