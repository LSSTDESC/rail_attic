import numpy as np
from scipy import stats
import qp
import utils


class Metrics:
    """ A superclass for metrics"""
    def __init__(self, pdfs=None, xvals=None, ztrue=None, name=None):
        """Class constructor.
        Parameters
        ----------
        pdfs: `ndarray`
            array of PDFS
        xvals: `ndarray`
            pdf bins (z grid)
        ztrue: `ndarray`
            true redshifts
        name: `str`
            the name of the metric
        """
        self._pdfs = pdfs
        self._xvals = xvals
        self._ztrue = ztrue
        self._name = name
        self._metric = None


    def evaluate(self):
        """
        Evaluates the metric a function of the truth and prediction

        Returns
        -------
        metric: `float` or `ndarray`
            value of the metric
        """
        print('No metric specified')
        return self._metric

    @property
    def metric(self):
        return self._metric



"""    Metrics subclasses below   """

class PIT(Metrics):
    """ Probability Integral Transform """
    def __init__(self, pdfs, xvals, ztrue, name="PIT"):
        """Class constructor. """
        super().__init__(pdfs, xvals, ztrue, name)
        self._qq = None

    def evaluate(self):
        """Compute PIT array using qp.Ensemble class"""
        pdfs_ensemble = qp.Ensemble(qp.interp, data=dict(xvals=self._xvals, yvals=self._pdfs))
        self._metric = np.array([pdfs_ensemble[i].cdf(self._ztrue[i])[0][0] for i in range(len(self._ztrue))])
        return self._metric

    @property
    def qq(self, n_quant=100):
        q_theory = np.linspace(0., 1., n_quant)
        q_data = np.quantile(self._metric, q_theory)
        self._qq = (q_theory, q_data)
        return self._qq

    def plot_pit_qq(self, bins=None, code=None, title=None, show_pit=True,
                    show_qq=True, pit_out_rate=None, savefig=False):
        """Make plot PIT-QQ as Figure 2 from Schmidt et al. 2020."""
        fig_filename = utils.plot_pit_qq(self, bins=bins, code=code, title=title,
                                         show_pit=show_pit, show_qq=show_qq,
                                         pit_out_rate=pit_out_rate,
                                         savefig=savefig)
        return fig_filename

class PitOutRate(Metrics):
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


class KS(Metrics):
    """ Kolmogorov-Smirnov statistic """
    def __init__(self, pdfs, xvals, ztrue, name="KS"):
        """Class constructor. """
        super().__init__(pdfs, xvals, ztrue, name)
        self._statistic = None
        self._pvalue = None
        self._pits = None

    def evaluate(self, pits=None):
        """ Use scipy.stats.kstest to compute the Kolmogorov-Smirnov statistic for
        the PIT values by comparing with a uniform distribution between 0 and 1. """
        if pits is None:
            pits = PIT(self._pdfs, self._xvals, self._ztrue).evaluate()
        self._pits = pits
        self._statistic, self._pvalue = stats.kstest(pits, stats.uniform.cdf)
        self._metric = self._statistic
        return self._statistic, self._pvalue

    @property
    def statistic(self):
        return self._statistic

    @property
    def pvalue(self):
        return self._pvalue

    def plot(self):
        utils.ks_plot(self)


class CvM(Metrics):
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


class AD(Metrics):
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


class CDE(Metrics):
    """ Conditional density loss """

    def __init__(self, pdfs, xvals, ztrue, name="CDE loss"):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(pdfs, xvals, ztrue, name)

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


class KLD(Metrics):
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



class CRPS(Metrics):
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
