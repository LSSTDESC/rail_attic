import numpy as np
from scipy import stats
import qp
from IPython.display import Markdown
import plots


class Metrics:
    """
       ***   Metrics class   ***
    Receives a Sample object as input.
    Computes PIT and QQ vectors on the initialization.
    It's the basis for the other metrics, such as KS, AD, and CvM.
    """
    def __init__(self, sample, n_quant=100, pit_min=0.0001, pit_max=0.9999, debug=False):
        """Class constructor
        Parameters
        ----------
        sample: `Sample`
            sample object defined in ./sample.py
        n_quant: `int`, (optional)
            number of quantiles for the QQ plot
        pit_min: `float`
            lower limit to define PIT outliers
            default is 0.0001
        pit_max:
            upper limit to define PIT outliers
            default is 0.9999
        """
        self._sample = sample
        self._n_quant = n_quant
        self._pit_min = pit_min
        self._pit_max = pit_max
        if debug:
            n = 1000
        else:
            n = len(self._sample)
        self._pit = np.array([self._sample._pdfs[i].cdf(self._sample._ztrue[i])[0][0]
                              for i in range(n)])
        Qtheory = np.linspace(0., 1., self.n_quant)
        Qdata = np.quantile(self._pit, Qtheory)
        self._qq_vectors = (Qtheory, Qdata)
        # Uniform distribution amplitude
        self._yscale_uniform = 1. / float(n_quant)
        # Distribution of PIT values as it was a PDF
        self._xvals = Qtheory
        self._pit_dist, self._pit_bins_edges = np.histogram(self._pit, bins=n_quant, density=True)
        self._uniform_dist = np.ones_like(self._pit_dist) * self._yscale_uniform
        # Define qp Ensemble to use CDF functionallity (an ensemble with only 1 PDF)
        self._pit_ensamble = qp.Ensemble(qp.hist, data=dict(bins=self._pit_bins_edges,
                                                            pdfs=np.array([self._pit_dist])))
        self._uniform_ensamble = qp.Ensemble(qp.interp, data=dict(xvals=self._xvals,
                                                                  yvals=np.array([self._uniform_dist])))
        self._pit_cdf = self._pit_ensamble.cdf(self._xvals)
        self._uniform_cdf = self._uniform_ensamble.cdf(self._xvals)
        # Fraction of outliers
        pit_n_outliers = len(self._pit[(self._pit < pit_min) | (self._pit > pit_max)])
        self._pit_out_rate = float(pit_n_outliers) / float(len(self._pit))

        # placeholders for metrics to be calculated
        self._cde_loss = None
        self._ks_stat = None
        self._ks_pvalue = None
        self._cvm_stat = None
        self._cvm_pvalue = None
        self._ad_stat = None
        self._ad_critical_values = None
        self._ad_significance_levels = None

    @property
    def sample(self):
        return self._sample

    @property
    def n_quant(self):
        return self._n_quant

    @property
    def pit(self):
        return self._pit

    @property
    def qq_vectors(self):
        return self._qq_vectors

    @property
    def yscale_uniform(self):
        return self._yscale_uniform

    @property
    def pit_out_rate(self):
        return self._pit_out_rate

    def plot_pit_qq(self, bins=None, label=None, title=None, show_pit=True,
                    show_qq=True, show_pit_out_rate=True, savefig=False):
        fig_filename = plots.plot_pit_qq(self, bins=bins, label=label, title=title,
                                         show_pit=show_pit, show_qq=show_qq,
                                         show_pit_out_rate=show_pit_out_rate,
                                         savefig=savefig)
        return fig_filename

    @property
    def cde_loss(self):
        return self._cde_loss

    @property
    def ks_stat(self):
        return self._ks_stat

    @property
    def ks_pvalue(self):
        return self._ks_pvalue

    @property
    def cvm_stat(self):
        return self._cvm_stat

    @property
    def cvm_pvalue(self):
        return self._cvm_pvalue

    @property
    def ad_stat(self):
        return self._ad_stat

    @property
    def ad_critical_values(self):
        return self._ad_critical_values

    @property
    def ad_significance_levels(self):
        return self._ad_significance_levels

    def compute_stats(self):
        self._cde_loss = CDE(self._sample)._cde_loss
        self._ks_stat = KS(self._sample).stat
        self._cvm_stat = CvM(self._sample).stat
        self._ad_stat = AD(self._sample).stat

    def markdown_summary_metrics(self, show_dc1=False):
        self.compute_stats()
        if show_dc1:
            metrics_table = str("|Metric|Value| DC1 ref. value|\n" +
                                "|---|---|---|\n" +
                                f"PIT out rate | {self._pit_out_rate:8.4f} | 0.0202 \n" +
                                f"CDE loss     | {self._cde_loss:8.4f} |  -10.60 \n" +
                                f"KS           | {self._ks_stat:8.4f} | 0.01294 \n" +
                                f"CvM          | {self._cvm_stat:8.4f} | 19.7154 \n" +
                                f"AD           | {self._ad_stat:8.4f} | 303.6519 ")
        else:
            metrics_table = str("|Metric|Value|\n" +
                                "|---|---|\n" +
                                f"PIT out rate | {self._pit_out_rate:8.4f}\n" +
                                f"CDE loss     | {self._cde_loss:8.4f}\n" +
                                f"KS           | {self._ks_stat:8.4f}\n" +
                                f"CvM          | {self._cvm_stat:8.4f}\n" +
                                f"AD           | {self._ad_stat:8.4f}")

        return Markdown(metrics_table)

    def print_summary_metrics(self):
        self.compute_stats()
        metrics_table = str(  # f"### {self._sample._name}\n" +
            "   Metric    |   Value \n" +
            "-------------|----------\n" +
            f"PIT out rate | {self._pit_out_rate:8.4f}\n" +
            f"CDE loss     | {self._cde_loss:8.4f}\n" +
            f"KS           | {self._ks_stat:8.4f}\n" +
            f"CvM          | {self._cvm_stat:8.4f}\n" +
            f"AD           | {self._ad_stat:8.4f}")
        print(metrics_table)
        return metrics_table

    #@classmethod
    #def dc1(cls): #, metric=None, code=None):
    #    cls.dc1 = DC1().results
    #    return cls.dc1
    @property
    def dc1(self):
        return DC1().results

    # def dc1_metrics_table(cls, dc1, code):
    #     metrics_table = str("|Metric|Value|\n|---|---|\n " +
    #                         f"PIT out rate | {dc1["PIT out rate"][f"{code}"]}\n CDE loss | -10.60\n KS | ???\n CvM | ???\n AD | ???")
    #     return Markdown(metrics_table)


class DC1:

    def __init__(self):
        # Reference values:
        self.codes = ("ANNz2", "BPZ", "CMNN", "Delight", "EAZY", "FlexZBoost",
                 "GPz", "LePhare", "METAPhoR", "SkyNet", "TPZ")
        self.metrics = ("PIT out rate", "CDE loss", "KS", "CvM", "AD")
        self.pit_out_rate = [0.0265, 0.0192, 0.0034, 0.0006, 0.0154, 0.0202,
                             0.0058, 0.0486, 0.0229, 0.0001, 0.0130]
        self.cde_loss = [-6.88, -7.82, -10.43, -8.33, -7.07, -10.60,
                         -9.93, -1.66, -6.28, -7.89, -9.55]
        self.ks = [0.01740478, 0.01118018, 0.00502691, 0.02396731, 0.04302462, 0.01294894,
                   0.01452443, 0.02449423, 0.02965564, 0.04911712, 0.00954685]
        self.cvm = [60.33973412, 37.09194799, 2.9165108, 105.65338329, 440.07007555, 19.71544373,
                    61.60230833, 141.08468956, 153.05291971, 961.53956815, 24.30815299]
        self.ad = [564.01888766, 358.09533373, 30.64646869, 624.17799304, 2000.11675363, 303.65198293,
                   618.63599149, 1212.07245582, 1445.53118933, 5689.32253132, 282.36983696]
    @property
    def results(self):
        results = {"PIT out rate": dict([(code, value) for code, value in zip(self.codes, self.pit_out_rate)]),
               "CDE loss": dict([(code, value) for code, value in zip(self.codes, self.cde_loss)]),
               "KS": dict([(code, value) for code, value in zip(self.codes, self.ks)]),
               "CvM": dict([(code, value) for code, value in zip(self.codes, self.cvm)]),
               "AD": dict([(code, value) for code, value in zip(self.codes, self.ad)])}
        return results






class CDE:
    """Computes the estimated conditional density loss described in
    Izbicki & Lee 2017 (arXiv:1704.08095).
    Parameters
    ----------
    sample: `Sample`
        sample object defined in ./sample.py
    """

    def __init__(self, sample):
        zgrid = sample._zgrid
        ztrue = sample._ztrue
        pdf = sample._pdfs.pdf([sample._zgrid])
        n_obs, n_grid = (pdf).shape
        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(np.array(pdf) ** 2, zgrid))
        # z bin closest to ztrue
        nns = [np.argmin(np.abs(zgrid - z)) for z in ztrue]
        # Calculate second term E[f*(Z | X)]
        term2 = np.mean(pdf[range(n_obs), nns])
        self._cde_loss = term1 - 2 * term2

    @property
    def cde_loss(self):
        return self._cde_loss


class KS:
    """
    Compute the Kolmogorov-Smirnov statistic and p-value for the PIT
    values by comparing with a uniform distribution between 0 and 1.
    Parameters
    ----------
    pit: `numpy.ndarray`
        array with PIT values for all galaxies in the sample
    """
    def __init__(self, metrics):
        self._metrics = metrics
        self._stat = np.max(np.abs(metrics._pit_cdf - metrics._uniform_cdf))
        self._bin_stat = np.argmax(np.abs(metrics._pit_cdf - metrics._uniform_cdf))
        self._pvalue = None
        # update Metrics object
        if self._metrics._ks_stat == None:
            self._metrics._ks_stat = float(self._stat)

    def plot(self):
        plots.ks_plot(self)

    @property
    def stat(self):
        return self._stat

    @property
    def pvalue(self):
        return self._pvalue


class CvM:
    """
    Compute the Cramer-von Mises statistic and p-value for the PIT values
    by comparing with a uniform distribution between 0 and 1.
    Parameters
    ----------
    pit: `numpy.ndarray`
        array with PIT values for all galaxies in the sample
    """

    def __init__(self, pit):
        self._pit = pit
        cvm_result = stats.cramervonmises(self._pit, "uniform")
        self._stat, self._pvalue = cvm_result.statistic, cvm_result.pvalue

    @property
    def stat(self):
        return self._stat

    @property
    def pvalue(self):
        return self._pvalue


class AD:
    """
    Compute the Anderson-Darling statistic and p-value for the PIT
    values by comparing with a uniform distribution between 0 and 1.
    Since the statistic diverges at 0 and 1, PIT values too close to
    0 or 1 are discarded.
    Parameters
    ----------
    pit: `numpy.ndarray`
        array with PIT values for all galaxies in the sample
    ad_pit_min, ad_pit_max: floats
        PIT values outside this range are discarded
    """

    def __init__(self, pit, ad_pit_min=0.001, ad_pit_max=0.999):
        mask = (pit > ad_pit_min) & (pit < ad_pit_max)
        self._stat, self._critical_values, self._significance_levels = stats.anderson(pit[mask])

    @property
    def stat(self):
        return self._stat

    @property
    def critical_values(self):
        return self._critical_values

    @property
    def significance_levels(self):
        return self._significance_levels


class CRPS:
    ''' = continuous rank probability score (Gneiting et al., 2006)'''

    def __init__(self):
        raise NotImplementedError
