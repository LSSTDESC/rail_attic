import numpy as np
from scipy import stats
import qp
import utils


class Metrics:
    """ A superclass for metrics"""

    def __init__(self, sample=None, name=None):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        self.sample = sample
        self.name = name

    def evaluate(self):
        """
        Evaluates the metric a function of the truth and prediction
        Parameters
        ----------
        data: `ndarray`
            PDFs or PITs, depending on the metric
        Returns
        -------
        metric: float
            value of the metric
        """
        print('No metric specified')
        metric = None
        return metric


# -------------------------------#
"""    Metrics subclasses    """


class PitOutRate(Metrics):
    """ Fraction of PIT outliers """

    def __init__(self, sample, name="PIT out rate"):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(sample, name)
        self._metric = None

    def evaluate(self, pit_min=0.0001, pit_max=0.9999):
        """Compute fraction of PIT outliers"""
        pits = self.sample.pit
        n_outliers = len(pits[(pits < pit_min) | (pits > pit_max)])
        self._metric = float(n_outliers) / float(len(self.sample))
        return self._metric

    @property
    def metric(self):
        return self._metric


class KS(Metrics):
    """ Kolmogorov-Smirnov statistic """

    def __init__(self, sample, name="KS"):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(sample, name)
        self._metric = None
        self._statistic = None
        self._pvalue = None

    def evaluate(self):
        """ Use scipy.stats.kstest to compute the Kolmogorov-Smirnov statistic for
        the PIT values by comparing with a uniform distribution between 0 and 1. """
        pits = self.sample.pit
        self._statistic, self._pvalue = stats.kstest(pits, stats.uniform.cdf)
        self._metric = self._statistic
        return self._statistic, self._pvalue

    @property
    def metric(self):
        return self._metric

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

    def __init__(self, sample, name="CvM"):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(sample, name)
        self._metric = None
        self._statistic = None
        self._pvalue = None

    def evaluate(self):
        """ Use scipy.stats.cramervonmises to compute the Cramer-von Mises statistic for
        the PIT values by comparing with a uniform distribution between 0 and 1. """
        pits = self.sample.pit
        cvm_result = stats.cramervonmises(pits, "uniform")
        self._statistic, self._pvalue = cvm_result.statistic, cvm_result.pvalue
        self._metric = self._statistic
        return self._statistic, self._pvalue

    @property
    def metric(self):
        return self._metric

    @property
    def statistic(self):
        return self._statistic

    @property
    def pvalue(self):
        return self._pvalue


class AD(Metrics):
    """ Anderson-Darling statistic """

    def __init__(self, sample, name="AD"):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(sample, name)
        self._metric = None
        self._statistic = None
        self._critical_values = None
        self._significance_level = None

    def evaluate(self, ad_pit_min=0.0, ad_pit_max=1.0):
        """ Use scipy.stats.anderson_ksamp to compute the Anderson-Darling statistic
        for the PIT values by comparing with a uniform distribution between 0 and 1.
        Up to the current version (1.6.2), scipy.stats.anderson does not support
        uniform distributions as reference for 1-sample test.

        Parameters
        ----------
        ad_pit_min, ad_pit_max: floats
            PIT values outside this range are discarded
        """
        pits = self.sample.pit
        uniform = np.arange(len(pits))
        mask = (pits > ad_pit_min) & (pits < ad_pit_max)
        ad_results = stats.anderson_ksamp([pits[mask], uniform[mask]])
        self._statistic, self._critical_values, self._significance_level = ad_results
        self._metric = self._statistic
        return self._statistic, self._critical_values, self._significance_level

    @property
    def metric(self):
        return self._metric

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

    def __init__(self, sample, name="CDE loss"):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(sample, name)
        self._metric = None

    def evaluate(self):
        """Evaluate the estimated conditional density loss described in
        Izbicki & Lee 2017 (arXiv:1704.08095). """
        pdf = self.sample.pdf([self.sample.zgrid])
        n_obs, n_grid = (pdf).shape
        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(np.array(pdf) ** 2, x=self.sample.zgrid))
        # z bin closest to ztrue
        nns = [np.argmin(np.abs(self.sample.zgrid - z)) for z in self.sample.ztrue]
        # Calculate second term E[f*(Z | X)]
        term2 = np.mean(pdf[range(n_obs), nns])
        self._metric = term1 - 2 * term2
        return self._metric

    @property
    def metric(self):
        return self._metric


class KLD(Metrics):
    """ Kullback-Leibler Divergence """

    def __init__(self, sample, name="KLD"):
        """Class constructor.
        Parameters
        ----------
        sample: `qp.ensemble`
            ensemble of PDFS
        name: `str`
            the name of the metric
        """
        super().__init__(sample, name)
        self._metric = None

    def evaluate(self):
        """ Use scipy.stats.entropy to compute the Kullback-Leibler
        Divergence between the empirical PIT distribution and a
        theoretical uniform distribution between 0 and 1."""
        pits = self.sample.pit
        xvals = self.sample.qq[0]
        pit_pdf, _ = np.histogram(pits, bins=len(xvals))
        uniform_pdf = np.full(self.sample.n_quant, 1.0 / float(self.sample.n_quant))
        self.S = stats.entropy(pit_pdf, uniform_pdf)
        self._metric = self.S
        return self._metric

    @property
    def metric(self):
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
        self._metric = None

    def evaluate(self):
        raise NotImplementedError
