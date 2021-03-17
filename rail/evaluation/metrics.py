import numpy as np
from scipy import stats
import qp
from IPython.display import Markdown
import utils


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
        if debug: # subset for quick tests
            n = 1000
        else:
            n = len(self._sample)
        self._pit = np.nan_to_num([self._sample._pdfs[i].cdf(self._sample._ztrue[i])[0][0]
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
        self._pit_cdf = self._pit_ensamble.cdf(self._xvals)[0]
        self._uniform_cdf = self._uniform_ensamble.cdf(self._xvals)[0]
        # Fraction of outliers
        pit_n_outliers = len(self._pit[(self._pit < pit_min) | (self._pit > pit_max)])
        self._pit_out_rate = float(pit_n_outliers) / float(len(self._pit))

        # placeholders for metrics to be calculated
        self._ks_stat = None
        self._ks_pvalue = None
        self._cvm_stat = None
        self._cvm_pvalue = None
        self._ad_stat = None
        self._ad_critical_values = None
        self._ad_significance_levels = None
        self._cde_loss = None


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
        fig_filename = utils.plot_pit_qq(self, bins=bins, label=label, title=title,
                                         show_pit=show_pit, show_qq=show_qq,
                                         show_pit_out_rate=show_pit_out_rate,
                                         savefig=savefig)
        return fig_filename


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

    @property
    def cde_loss_stat(self):
        return self._cde_loss


    def compute_stats(self):
        #if self._ks_stat is None:
        self._ks_stat = KS(self).stat
        #if self._cvm_stat is None:
        self._cvm_stat = CvM(self).stat
        #if self._ad_stat is None:
        self._ad_stat = AD(self).stat
        #if self._cde_loss is None:
        self._cde_loss = CDE(self)._cde_loss

    def markdown_table(self, show_dc1=False):
        self.compute_stats()
        if show_dc1:
            dc1 = self.dc1
            table = str("Metric|Value|DC1 reference value \n ---|---:|---: \n ")
            table += f"PIT out rate | {self._pit_out_rate:11.4f} |{dc1['PIT out rate'][self._sample._code]:11.4f} \n"
            table += f"KS           | {self._ks_stat:11.4f} |{dc1['KS'][self._sample._code]:11.4f} \n"
            table += f"CvM          | {self._cvm_stat:11.4f} |{dc1['CvM'][self._sample._code]:11.4f} \n"
            table += f"AD           | {self._ad_stat:11.4f} |{dc1['AD'][self._sample._code]:11.4f} \n"
            table += f"CDE loss     | {self._cde_loss:11.4f} |{dc1['CDE loss'][self._sample._code]:11.4f}"
        else:
            table = "Metric|Value \n ---|---: \n "
            table += f"PIT out rate | {self._pit_out_rate:11.4f} \n"
            table += f"KS           | {self._ks_stat:11.4f} \n"
            table += f"CvM          | {self._cvm_stat:11.4f}\n"
            table += f"AD           | {self._ad_stat:11.4f}\n"
            table += f"CDE loss     | {self._cde_loss:11.4f}\n"
        return Markdown(table)

    def print_table(self):
        self.compute_stats()
        table = str(
             "   Metric    |    Value \n" +
             "-------------|-------------\n" +
            f"PIT out rate | {self._pit_out_rate:11.4f}\n" +
            f"KS           | {self._ks_stat:11.4f}\n" +
            f"CvM          | {self._cvm_stat:11.4f}\n" +
            f"AD           | {self._ad_stat:11.4f}\n" +
            f"CDE loss     | {self._cde_loss:11.4f}")
        print(table)

    @property
    def dc1(self):
        return DC1().results






class CDE:
    """Computes the estimated conditional density loss described in
    Izbicki & Lee 2017 (arXiv:1704.08095).
    Parameters
    ----------
    sample: `Sample`
        sample object defined in ./sample.py
    """

    def __init__(self, metrics):
        sample = metrics._sample
        pdf = sample._pdfs.pdf([sample._zgrid])
        n_obs, n_grid = (pdf).shape
        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(np.array(pdf) ** 2, sample._zgrid))
        # z bin closest to ztrue
        nns = [np.argmin(np.abs(sample._zgrid - z)) for z in sample._ztrue]
        # Calculate second term E[f*(Z | X)]
        term2 = np.mean(pdf[range(n_obs), nns])
        self._cde_loss = term1 - 2 * term2
        # update Metrics object
        metrics._cde_loss = self._cde_loss

    @property
    def cde_loss(self):
        return self._cde_loss


class KS:
    """
    Compute the Kolmogorov-Smirnov statistic and p-value (TBD) for the PIT
    values by comparing with a uniform distribution between 0 and 1.

    Parameters
    ----------
    metrics: `metrics` object
        instance of metrics base class which is connected to a given sample
    """
    def __init__(self, metrics, scipy=False):
        self._metrics = metrics
        if scipy:
            self._stat, self._pvalue = stats.kstest(metrics._pit, "uniform")
        else:
            self._stat, self._pvalue = np.max(np.abs(metrics._pit_cdf - metrics._uniform_cdf)), None # p=value TBD
        # update Metrics object
        metrics._ks_stat = self._stat

    def plot(self):
        utils.ks_plot(self)

    @property
    def stat(self):
        return self._stat

    @property
    def pvalue(self):
        return self._pvalue


class CvM:
    """
    Compute the Cramer-von Mises statistic and p-value (TBD) for the PIT
    values by comparing with a uniform distribution between 0 and 1.

    Parameters
    ----------
    metrics: `metrics` object
        instance of metrics base class which is connected to a given sample
    """

    def __init__(self, metrics, scipy=False):
        if scipy:
            cvm_result = stats.cramervonmises(metrics._pit, "uniform")
            self._stat, self._pvalue = cvm_result.statistic, cvm_result.pvalue
        else:
            self._stat, self._pvalue = np.sqrt(np.trapz((metrics._pit_cdf - metrics._uniform_cdf)**2, metrics._uniform_cdf)), None
        # update Metrics object
        metrics._cvm_stat = self._stat

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

    def __init__(self, metrics, ad_pit_min=0.0, ad_pit_max=1.0, scipy=False):

        mask_pit = (metrics._pit >= ad_pit_min) & (metrics._pit  <= ad_pit_max)
        if (ad_pit_min != 0.0) or (ad_pit_max != 1.0):
            n_out = len(metrics._pit) - len(metrics._pit[mask_pit])
            perc_out = (float(n_out)/float(len(metrics._pit)))*100.
            print(f"{n_out} outliers (PIT<{ad_pit_min} or PIT>{ad_pit_max}) removed from the calculation ({perc_out:.1f}%)")
        if scipy:
            #self._stat, self._critical_values, self._significance_levels = stats.anderson(metrics._pit[mask_pit])
            self._stat, self._critical_values, self._significance_levels = None, None, None
            print("Comparison to uniform distribution is not available in scipy.stats.anderson method.")
        else:
            ad_xvals = np.linspace(ad_pit_min, ad_pit_max, metrics._n_quant)
            ad_yscale_uniform = (ad_pit_max-ad_pit_min)/float(metrics._n_quant)
            ad_pit_dist, ad_pit_bins_edges = np.histogram(metrics._pit[mask_pit], bins=metrics._n_quant, density=True)
            ad_uniform_dist = np.ones_like(ad_pit_dist) * ad_yscale_uniform
            # Redo CDFs to consider outliers mask
            ad_pit_ensamble = qp.Ensemble(qp.hist, data=dict(bins=ad_pit_bins_edges, pdfs=np.array([ad_pit_dist])))
            ad_pit_cdf = ad_pit_ensamble.cdf(ad_xvals)[0]
            ad_uniform_ensamble = qp.Ensemble(qp.hist,
                                              data=dict(bins=ad_pit_bins_edges, pdfs=np.array([ad_uniform_dist])))
            ad_uniform_cdf = ad_uniform_ensamble.cdf(ad_xvals)[0]
            numerator = ((ad_pit_cdf - ad_uniform_cdf)**2)
            denominator = (ad_uniform_cdf*(1.-ad_uniform_cdf))
            with np.errstate(divide='ignore', invalid='ignore'):
                self._stat = np.sqrt(float(len(metrics._sample)) * np.trapz(np.nan_to_num(numerator/denominator), ad_uniform_cdf))
            self._critical_values = None
            self._significance_levels = None

        # update Metrics object
        metrics._ad_stat = self._stat

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
