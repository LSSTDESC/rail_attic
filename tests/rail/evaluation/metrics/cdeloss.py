import numpy as np
from .base import MetricEvaluator
from rail.evaluation.utils import stat_and_pval


class CDELoss(MetricEvaluator):
    """ Conditional density loss """
    def __init__(self, qp_ens, zgrid, ztrue):
        """Class constructor"""
        super().__init__(qp_ens)

        self._pdfs = qp_ens.pdf(zgrid)
        self._xvals = zgrid
        self._ztrue = ztrue
        self._npdf = qp_ens.npdf

    def evaluate(self):
        """Evaluate the estimated conditional density loss described in
        Izbicki & Lee 2017 (arXiv:1704.08095).

        Notes
        -----
        """

        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(self._pdfs ** 2, x=self._xvals))
        # z bin closest to ztrue
        nns = [np.argmin(np.abs(self._xvals - z)) for z in self._ztrue]
        # Calculate second term E[f*(Z | X)]
        term2 = np.mean(self._pdfs[range(self._npdf), nns])
        cdeloss = term1 - 2 * term2
        return stat_and_pval(cdeloss, np.nan)
