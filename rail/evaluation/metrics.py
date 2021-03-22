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
        self._pit = np.nan_to_num([self._sample._pdfs[i].cdf(self._sample._ztrue[i])[0][0] for i in range(n)])
        # Quantiles
        Qtheory = np.linspace(0., 1., self.n_quant)
        Qdata = np.quantile(self._pit, Qtheory)
        self._qq_vectors = (Qtheory, Qdata)
        # Normalized distribution of PIT values (PIT PDF)
        self._xvals = Qtheory
        self._pit_pdf, self._pit_bins_edges = np.histogram(self._pit, bins=n_quant, density=True)
        self._uniform_pdf = np.full(n_quant, 1.0 / float(n_quant))
        # Define qp Ensemble to use CDF functionallity (an ensemble with only 1 PDF)
        self._pit_ensamble = qp.Ensemble(qp.hist, data=dict(bins=self._pit_bins_edges,
                                                            pdfs=np.array([self._pit_pdf])))
        self._uniform_ensamble = qp.Ensemble(qp.interp, data=dict(xvals=self._xvals,
                                                                  yvals=np.array([self._uniform_pdf])))
        self._pit_cdf = self._pit_ensamble.cdf(self._xvals)[0]
        self._uniform_cdf = self._uniform_ensamble.cdf(self._xvals)[0]

        # placeholders for metrics to be calculated
        self._pit_out_rate = None
        self._cde_loss = None
        self._kld = None
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
    def pit_min(self):
        return self._pit_min

    @property
    def pit_max(self):
        return self._pit_max

    @property
    def pit(self):
        return self._pit

    @property
    def pit_pdf(self):
        return self._pit_pdf

    @property
    def uniform_pdf(self):
        return self._uniform_pdf

    @property
    def qq_vectors(self):
        return self._qq_vectors

    @property
    def pit_out_rate(self):
        return self._pit_out_rate

    @property
    def cde_loss(self):
        return self._cde_loss

    @property
    def kld(self):
        return self._kld

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
    def dc1(self):
        return utils.DC1().results




    def plot_pit_qq(self, bins=None, label=None, title=None, show_pit=True,
                    show_qq=True, show_pit_out_rate=True, savefig=False):
        """Make plot PIT-QQ as Figure 2 from Schmidt et al. 2020."""
        fig_filename = utils.plot_pit_qq(self, bins=bins, label=label, title=title,
                                         show_pit=show_pit, show_qq=show_qq,
                                         show_pit_out_rate=show_pit_out_rate,
                                         savefig=savefig)
        return fig_filename


    def compute_metrics(self):
        self._pit_out_rate = PitOutRate(self)._pit_out_rate
        self._cde_loss = CDE(self)._cde_loss
        #self._kld = KLD(self).kld
        self._ks_stat = KS(self).stat
        self._cvm_stat = CvM(self).stat
        self._ad_stat = AD(self).stat

    def markdown_metrics_table(self, show_dc1=False):
        self.compute_metrics()
        if show_dc1:
            dc1 = self.dc1
            table = str("Metric|Value|DC1 reference value \n ---|---:|---: \n ")
            table += f"PIT out rate | {self._pit_out_rate:11.4f} |{dc1['PIT out rate'][self._sample._code]:11.4f} \n"
            table += f"CDE loss     | {self._cde_loss:11.4f} |{dc1['CDE loss'][self._sample._code]:11.4f} \n"
            #table += f"KLD          | {self._kld:11.4f}      |  N/A  \n"
            table += f"KS           | {self._ks_stat:11.4f}  |{dc1['KS'][self._sample._code]:11.4f} \n"
            table += f"CvM          | {self._cvm_stat:11.4f} |{dc1['CvM'][self._sample._code]:11.4f} \n"
            table += f"AD           | {self._ad_stat:11.4f}  |{dc1['AD'][self._sample._code]:11.4f} \n"
        else:
            table = "Metric|Value \n ---|---: \n "
            table += f"PIT out rate | {self._pit_out_rate:11.4f} \n"
            table += f"CDE loss     | {self._cde_loss:11.4f}\n"
            #table += f"KLD          | {self._kld:11.4f}      |  N/A  \n"
            table += f"KS           | {self._ks_stat:11.4f} \n"
            table += f"CvM          | {self._cvm_stat:11.4f}\n"
            table += f"AD           | {self._ad_stat:11.4f}\n"
        return Markdown(table)

    def print_metrics_table(self):
        self.compute_metrics()
        table = str(
             "   Metric    |    Value \n" +
             "-------------|-------------\n" +
            f"PIT out rate | {self._pit_out_rate:11.4f}\n" +
            f"CDE loss     | {self._cde_loss:11.4f}\n " +
            #f"KLD          | {self._kld:11.4f}\n" +
            f"KS           | {self._ks_stat:11.4f}\n" +
            f"CvM          | {self._cvm_stat:11.4f}\n" +
            f"AD           | {self._ad_stat:11.4f}\n")
        print(table)




class PitOutRate:
    """ Fraction of PIT outliers """
    def __init__(self, metrics):
        """Class constructor.
        Compute fraction of PIT values which are close to 0 (pit < pit_min) and 1 (pit > pit_max).
        pit_min and pit_max limits are parameters of the metrics parent object.

        Parameters
        ----------
        metrics: `metrics` object
            instance of metrics base class which is connected to a given sample
        """
        pit_n_outliers = len(metrics.pit[(metrics.pit < metrics.pit_min) | (metrics.pit > metrics.pit_max)])
        self._pit_out_rate = float(pit_n_outliers) / float(len(metrics.pit))
        metrics._pit_out_rate = self._pit_out_rate

    @property
    def pit_out_rate(self):
        return self._pit_out_rate


class CDE:
    """Computes the estimated conditional density loss described in
    Izbicki & Lee 2017 (arXiv:1704.08095). """

    def __init__(self, metrics):
        """Class constructor.
        Compute CDE loss statistic and update the
        parent metric object with property metrics._cde_loss .

        Parameters
        ----------
        metrics: `metrics` object
            instance of metrics base class which is connected to a given sample
        """
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
    def __init__(self, metrics, scipy=True):
        self._metrics = metrics
        if scipy:
            self._stat, self._pvalue = stats.kstest(metrics.pit, "uniform")
        else:
            self._stat, self._pvalue = np.max(np.abs(metrics._pit_cdf - metrics._uniform_cdf)), None  # p-value TBD
        self._metrics = metrics
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

    def __init__(self, metrics, scipy=True):
        if scipy:
            cvm_result = stats.cramervonmises(metrics._pit, "uniform")
            self._stat, self._pvalue = cvm_result.statistic, cvm_result.pvalue
        else:
            self._stat, self._pvalue = np.sqrt(np.trapz((metrics._pit_cdf - metrics._uniform_cdf) ** 2, metrics._uniform_cdf)), None

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

    def __init__(self, metrics, ad_pit_min=0.0, ad_pit_max=1.0):

        mask_pit = (metrics._pit >= ad_pit_min) & (metrics._pit  <= ad_pit_max)
        if (ad_pit_min != 0.0) or (ad_pit_max != 1.0):
            n_out = len(metrics._pit) - len(metrics._pit[mask_pit])
            perc_out = (float(n_out)/float(len(metrics._pit)))*100.
            print(f"{n_out} outliers (PIT<{ad_pit_min} or PIT>{ad_pit_max}) removed from the calculation ({perc_out:.1f}%)")

        ad_xvals = np.linspace(ad_pit_min, ad_pit_max, metrics.n_quant)
        ad_yscale_uniform = (ad_pit_max-ad_pit_min)/float(metrics._n_quant)
        ad_pit_dist, ad_pit_bins_edges = np.histogram(metrics.pit[mask_pit], bins=metrics.n_quant, density=True)
        ad_uniform_dist = np.full(metrics.n_quant, ad_yscale_uniform)
        # Redo CDFs to account for outliers mask
        ad_pit_ensamble = qp.Ensemble(qp.hist, data=dict(bins=ad_pit_bins_edges, pdfs=np.array([ad_pit_dist])))
        ad_pit_cdf = ad_pit_ensamble.cdf(ad_xvals)[0]
        ad_uniform_ensamble = qp.Ensemble(qp.hist,
                                          data=dict(bins=ad_pit_bins_edges, pdfs=np.array([ad_uniform_dist])))
        ad_uniform_cdf = ad_uniform_ensamble.cdf(ad_xvals)[0]
        numerator = ((ad_pit_cdf - ad_uniform_cdf)**2)
        denominator = (ad_uniform_cdf*(1.-ad_uniform_cdf))
        with np.errstate(divide='ignore', invalid='ignore'):
            self._stat = np.sqrt(float(len(metrics._sample)) * np.trapz(np.nan_to_num(numerator/denominator), ad_uniform_cdf))

        # update Metrics object
        metrics._ad_stat = self._stat

    @property
    def stat(self):
        return self._stat




class KLD:
    """
    Compute the Kullback-Leibler Divergence between the the empirical PIT
    distribution and a theoretical uniform distribution between 0 and 1."""

    def __init__(self, metrics):
        """Class constructor.
        Compute KLD statistic using scipy.stats.entropy and update
        the parent metric object with property metrics._kld_stat .

        Parameters
        ----------
        metrics: `metrics` object
            instance of metrics base class which is connected to a given sample
        """
        self._kld = stats.entropy(metrics.pit_pdf, metrics.uniform_pdf)
        # update Metrics object
        metrics._kld = self._kld

    @property
    def kld(self):
        return self._kld


class CRPS:
    ''' = continuous rank probability score (Gneiting et al., 2006)'''

    def __init__(self):
        raise NotImplementedError
