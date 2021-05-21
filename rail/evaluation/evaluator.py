import numpy as np
from scipy import stats

import qp
import plot_utils
from IPython.display import Markdown
# from rail.evaluation.qp_metrics import KS, CvM



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


    def evaluate(self):
        """
        Evaluates the metric a function of the truth and prediction

        Returns
        -------
        metric: dictionary
            value of the metric and statistics thereof
        """
        raise NotImplementedError



"""    Metrics subclasses below   """




class CDE(Evaluator):
    """ Conditional density loss """

    def __init__(self, qp_ens, ztrue):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(qp_ens, ztrue, name)

    def evaluate(self):
        """Evaluate the estimated conditional density loss described in
        Izbicki & Lee 2017 (arXiv:1704.08095). """
        pdf = self._pdfs# self.sample.pdf([self.sample.zgrid])
        n_obs, n_grid = (pdf).shape
        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(np.array(pdf) ** 2, x=self._xvals))
        # z bin closest to ztrue
        nns = [np.argmin(np.abs(self._xvals - z)) for z in self._ztrue]
        # Calculate second term E[f*(Z | X)]
        term2 = np.mean(pdf[range(n_obs), nns])
        self._metric = term1 - 2 * term2
        return self._metric

    @property
    def metric(self):
        return self._metric



class CRPS(Evaluator):
    ''' Continuous rank probability score (Gneiting et al., 2006)'''

    def __init__(self, sample, name="CRPS"):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(sample, name)


    def evaluate(self):
        raise NotImplementedError
