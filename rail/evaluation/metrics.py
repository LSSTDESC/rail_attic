import numpy as np
from scipy import stats
import qp
import utils
from IPython.display import Markdown
from rail.evaluation.qp_metrics import KS, CvM

class Evaluator:
    """ A superclass for metrics evaluations"""
    def __init__(self, qp_ens):
        """Class constructor.
        Parameters
        ----------
        qp_obj: `qp object`
            PDF ensemble in qp format
        """
        
        self._qp_ens = qp_ens


    def evaluate(self):
        """
        Evaluates the metric a function of the truth and prediction

        Returns
        -------
        metric: `float` or `ndarray`
            value of the metric
        """
        raise NotImplementedError



"""    Metrics subclasses below   """

class PIT(Evaluator):
    """ Probability Integral Transform """
    def __init__(self, qp_ens, ztrue):
        """Class constructor. """
        super().__init__(qp_ens, ztrue)
        self._ztrue = ztrue
        self._pit = np.array([self._qp_ens[i].cdf(self._ztrue[i])[0][0] for i in range(len(self._ztrue))])

    def evaluate(self, eval_grid='None'):
        """Compute PIT array using qp.Ensemble class"""
        self.qq = _evaluate_qq(eval_grid)
        self.ks, self. ks_pval = KS(self._qp_ens, stats.uniform)
        self.cvm, self.cvm_pval = CvM(self._qp_ens, stats.uniform)

    def _evaluate_qq(self, eval_grid='None'):
        if eval_grid == 'None':
            eval_grid = np.linspace(0, 1, 100)
        q_theory = eval_grid
        q_data = np.quantile(self._pit, q_theory)
        qq = (q_theory, q_data)
        return qq

    def plot_all_pit(self):
        utils.ks_plot(self)


    def plot_pit_qq(self, bins=None, code=None, title=None, show_pit=True,
                    show_qq=True, pit_out_rate=None, savefig=False):
        """Make plot PIT-QQ as Figure 2 from Schmidt et al. 2020."""
        fig_filename = utils.plot_pit_qq(self, bins=bins, code=code, title=title,
                                         show_pit=show_pit, show_qq=show_qq,
                                         pit_out_rate=pit_out_rate,
                                         savefig=savefig)
        return fig_filename

class PitOutRate(Evaluator):
    """ Fraction of PIT outliers """
    def __init__(self, pdfs, xvals, ztrue, name="PIT out rate"):
        """Class constructor. """
        super().__init__(pdfs, xvals, ztrue, name)

    def evaluate(self, pits=None, pit_min=0.0001, pit_max=0.9999):
        """Compute fraction of PIT outliers"""
        if pits is None:
            pits = PIT(self._pdfs, self._xvals, self._ztrue).evaluate()
        n_outliers = len(pits[(pits < pit_min) | (pits > pit_max)])
        self._metric = float(n_outliers) / float(len(pits))
        return self._metric


class CvM(Evaluator):
    """ Cramer-von Mises statistic """

    def __init__(self, pdfs, xvals, ztrue, name="CvM"):
        """Class constructor. """
        super().__init__(pdfs, xvals, ztrue, name)
        self._statistic = None
        self._pvalue = None

    def evaluate(self, pits=None):
        """ Use scipy.stats.cramervonmises to compute the Cramer-von Mises statistic for
        the PIT values by comparing with a uniform distribution between 0 and 1. """
        if pits is None:
            pits = PIT(self._pdfs, self._xvals, self._ztrue).evaluate()
        cvm_result = stats.cramervonmises(pits, stats.uniform.cdf)
        self._statistic, self._pvalue = cvm_result.statistic, cvm_result.pvalue
        self._metric = self._statistic
        return self._statistic, self._pvalue

    @property
    def statistic(self):
        return self._statistic

    @property
    def pvalue(self):
        return self._pvalue


class AD(Evaluator):
    """ Anderson-Darling statistic """
    def __init__(self, pdfs, xvals, ztrue, name="AD"):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(pdfs, xvals, ztrue, name)
        self._statistic = None
        self._critical_values = None
        self._significance_level = None

    def evaluate(self, pits=None, ad_pit_min=0.0, ad_pit_max=1.0):
        """ Use scipy.stats.anderson_ksamp to compute the Anderson-Darling statistic
        for the PIT values by comparing with a uniform distribution between 0 and 1.
        Up to the current version (1.6.2), scipy.stats.anderson does not support
        uniform distributions as reference for 1-sample test.

        Parameters
        ----------
        ad_pit_min, ad_pit_max: floats
            PIT values outside this range are discarded
        """
        if pits is None:
            pits = PIT(self._pdfs, self._xvals, self._ztrue).evaluate()
        mask = (pits >= ad_pit_min) & (pits <= ad_pit_max)
        pits_clean = pits[mask]
        diff = len(pits) - len(pits_clean)
        if diff > 0:
            print(f"{diff} PITs removed from the sample.")
        uniform_yvals = np.linspace(ad_pit_min, ad_pit_max, len(pits_clean))
        ad_results = stats.anderson_ksamp([pits_clean, uniform_yvals])
        self._statistic, self._critical_values, self._significance_level = ad_results
        self._metric = self._statistic

        return self._statistic, self._critical_values, self._significance_level

    @property
    def statistic(self):
        return self._statistic

    @property
    def critical_values(self):
        return self._critical_values

    @property
    def significance_level(self):
        return self._significance_level


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


class KLD(Evaluator):
    """ Kullback-Leibler Divergence """

    def __init__(self, pdfs, xvals, ztrue, name="KLD"):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(pdfs, xvals, ztrue, name)

    def evaluate(self, pits=None):
        """ Use scipy.stats.entropy to compute the Kullback-Leibler
        Divergence between the empirical PIT distribution and a
        theoretical uniform distribution between 0 and 1."""
        if pits is None:
            pits = PIT(self._pdfs, self._xvals, self._ztrue).evaluate()
        uniform_yvals = np.linspace(0., 1., len(pits))
        pit_pdf, _ = np.histogram(pits, bins=len(self._xvals))
        uniform_pdf, _ = np.histogram(uniform_yvals, bins=len(self._xvals))

        self._metric = stats.entropy(pit_pdf, uniform_pdf)
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





class Summary:
    """ Summary tables with all metrics available. """
    def __init__(self, pdfs, xvals, ztrue):
        """Class constructor."""
        # placeholders for metrics to be calculated
        self._pdfs = pdfs
        self._xvals = xvals
        self._ztrue = ztrue
        self._pit_out_rate = None
        self._ks = None
        self._cvm = None
        self._ad = None
        self._kld = None
        self._cde_loss = None

    def evaluate_all(self, pits=None):
        if pits is None:
            pits = PIT(self._pdfs, self._xvals, self._ztrue).evaluate()
        self._pit_out_rate = PitOutRate(self._pdfs, self._xvals, self._ztrue).evaluate(pits=pits)
        self._ks, _ = KS(self._pdfs, self._xvals, self._ztrue).evaluate(pits=pits)
        self._cvm, _ = CvM(self._pdfs, self._xvals, self._ztrue).evaluate(pits=pits)
        self._ad, _, _ = AD(self._pdfs, self._xvals, self._ztrue).evaluate(pits=pits)
        self._kld = KLD(self._pdfs, self._xvals, self._ztrue).evaluate(pits=pits)
        self._cde_loss = CDE(self._pdfs, self._xvals, self._ztrue).evaluate()

    def markdown_metrics_table(self, show_dc1=None, pits=None):
        self.evaluate_all(pits=pits)
        if show_dc1:
            dc1 = utils.DC1()
            if show_dc1 not in dc1.codes:
                raise ValueError(f"{show_dc1} not in the list of codes from DC1: {dc1.codes}" )
            table = str("Metric|Value|DC1 reference value \n ---|---:|---: \n ")
            table += f"PIT out rate | {self._pit_out_rate:11.4f} |{dc1.results['PIT out rate'][show_dc1]:11.4f} \n"
            table += f"KS           | {self._ks:11.4f}  |{dc1.results['KS'][show_dc1]:11.4f} \n"
            table += f"CvM          | {self._cvm:11.4f} |{dc1.results['CvM'][show_dc1]:11.4f} \n"
            table += f"AD           | {self._ad:11.4f}  |{dc1.results['AD'][show_dc1]:11.4f} \n"
            table += f"KLD          | {self._kld:11.4f}      |  N/A  \n"
            table += f"CDE loss     | {self._cde_loss:11.2f} |{dc1.results['CDE loss'][show_dc1]:11.2f} \n"
        else:
            table = "Metric|Value \n ---|---: \n "
            table += f"PIT out rate | {self._pit_out_rate:11.4f} \n"
            table += f"KS           | {self._ks:11.4f}  \n"
            table += f"CvM          | {self._cvm:11.4f} \n"
            table += f"AD           | {self._ad:11.4f}  \n"
            table += f"CDE loss     | {self._cde_loss:11.2f} \n"
            table += f"KLD          | {self._kld:11.4f}      \n"
        return Markdown(table)

    def print_metrics_table(self, pits=None):
        self.evaluate_all(pits=pits)
        table = str(
            "   Metric    |    Value \n" +
            "-------------|-------------\n" +
            f"PIT out rate | {self._pit_out_rate:11.4f}\n" +
            f"KS           | {self._ks:11.4f}\n" +
            f"CvM          | {self._cvm:11.4f}\n" +
            f"AD           | {self._ad:11.4f}\n" +
            f"KLD          | {self._kld:11.4f}\n" +
            f"CDE loss     | {self._cde_loss:11.4f}\n" )
        print(table)
