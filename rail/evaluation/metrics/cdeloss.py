import inspect
import numpy as np
from scipy import stats
import qp
from rail.evaluation.evaluator import Evaluator

default_grid = np.linspace(0., 1., 100)

class CDELoss(Evaluator):
    """ Conditional density loss """
    def __init__(self, qp_ens, ztrue):
        """Class constructor"""
        super().__init__(qp_ens)

        self._ztrue = ztrue
        self._pdfs = qp_ens

    def evaluate(self, eval_grid=default_grid):
        """Evaluate the estimated conditional density loss described in
        Izbicki & Lee 2017 (arXiv:1704.08095).

        Notes
        -----
        Double-check that this is doing the right thing and that the types work out. . .
        """

        # first step here needs to be conversion to gridded parameterization
        grid_pdfs = self._pdfs.convert_to(qp.interp_gen, xvals=eval_grid)
        yvals = grid_pdfs.pdf(eval_grid)
        #  (or some simple transformation thereof?)

        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(ygrid ** 2, x=self._xvals))
        # z bin closest to ztrue
        nns = [np.argmin(np.abs(eval_grid - z)) for z in self._ztrue]
        # Calculate second term E[f*(Z | X)]
        term2 = np.mean(pdf[:, nns])
        cdeloss = term1 - 2 * term2
        return cdeloss
