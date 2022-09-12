import inspect
import numpy as np
from scipy import stats
import qp
from .base import MetricEvaluator
from rail.evaluation.utils import stat_and_pval, stat_crit_sig

default_quants = np.linspace(0, 1, 100)
_pitMetaMetrics = {}


def PITMetaMetric(cls):
    """Decorator function to attach metrics to a class"""
    argspec = inspect.getargspec(cls.evaluate)
    if argspec.defaults is not None:
        num_defaults = len(argspec.defaults)
        kwargs = dict(zip(argspec.args[-num_defaults:], argspec.defaults))
        _pitMetaMetrics.setdefault(cls, {})["default"] = kwargs
    return cls


class PIT(MetricEvaluator):
    """ Probability Integral Transform """

    def __init__(self, qp_ens, ztrue):
        """Class constructor"""
        super().__init__(qp_ens)

        self._ztrue = ztrue
        self._pit_samps = np.array([self._qp_ens[i].cdf(self._ztrue[i])[0][0] for i in range(len(self._ztrue))])


    @property
    def pit_samps(self):
        """Return the samples used to compute the PIT"""
        return self._pit_samps


    def evaluate(self, eval_grid=default_quants, meta_options=_pitMetaMetrics):
        """Compute PIT array using qp.Ensemble class
        Notes
        -----
        We will create a quantile Ensemble to store the PIT distribution, but also store the
        full set of PIT values as ancillary data of the (single PDF) ensemble.  I think
        the current metrics do not actually need the distribution, but we'll keep it here
        in case future PIT metrics need to make use of it.
        """
        n_pit = np.min([len(self._pit_samps), len(eval_grid)])
        if n_pit < len(eval_grid): #pragma: no cover
            eval_grid = np.linspace(0, 1, n_pit)
        data_quants = np.quantile(self._pit_samps, eval_grid)
        pit = qp.Ensemble(qp.quant_piecewise, data=dict(quants=eval_grid, locs=np.atleast_2d(data_quants)))

        #pit = qp.spline_from_samples(xvals=eval_grid,
        #                             samples=np.atleast_2d(self._pit_samps))
        #pit.samples = self._pit_samps

        if meta_options is not None:
            metamets = {}
            for cls, params in meta_options.items():
                meta = cls(self._pit_samps, pit)
                for name, kwargs in params.items():
                    metamets[(cls, name)] = meta.evaluate(**kwargs)

        # self.qq = _evaluate_qq(self._eval_grid)
        return pit, metamets

    # def _evaluate_qq(self):
    #     q_data = qp.convert(self._pit, 'quant', quants=self._eval_grid)
    #     return q_data
    #
    # def plot_all_pit(self):
    #     plot_utils.ks_plot(self)
    #
    #
    # def plot_pit_qq(self, bins=None, code=None, title=None, show_pit=True,
    #                 show_qq=True, pit_out_rate=None, savefig=False):
    #     """Make plot PIT-QQ as Figure 2 from Schmidt et al. 2020."""
    #     fig_filename = plot_utils.plot_pit_qq(self, bins=bins, code=code, title=title,
    #                                      show_pit=show_pit, show_qq=show_qq,
    #                                      pit_out_rate=pit_out_rate,
    #                                      savefig=savefig)
    #     return fig_filename


class PITMeta():
    """ A superclass for metrics of the PIT"""

    def __init__(self, pit_vals, pit):
        """Class constructor.
        Parameters
        ----------
        pit: qp.spline_from_samples object
            PIT values
        """
        self._pit = pit
        self._pit_vals = pit_vals

    # they all seem to have a way to trim the ends, so maybe just bring those together here?
    # def _trim(self, pit_min=0., pit_max=1.):
    #

    def evaluate(self): #pragma: no cover
        """
        Evaluates the metric a function of the truth and prediction

        Returns
        -------
        metric: dictionary
            value of the metric and statistics thereof
        """
        raise NotImplementedError


@PITMetaMetric
class PITOutRate(PITMeta):
    """ Fraction of PIT outliers """

    def evaluate(self, pit_min=0.0001, pit_max=0.9999):
        """Compute fraction of PIT outliers"""
        out_area = (self._pit.cdf(pit_min) + (1. - self._pit.cdf(pit_max)))[0][0]
        return out_area


@PITMetaMetric
class PITKS(PITMeta):
    """ Kolmogorov-Smirnov test statistic """

    def evaluate(self):
        """ Use scipy.stats.kstest to compute the Kolmogorov-Smirnov test statistic for
        the PIT values by comparing with a uniform distribution between 0 and 1. """
        stat, pval = stats.kstest(self._pit_vals, 'uniform')
        return stat_and_pval(stat, pval)


@PITMetaMetric
class PITCvM(PITMeta):
    """ Cramer-von Mises statistic """

    def evaluate(self):
        """ Use scipy.stats.cramervonmises to compute the Cramer-von Mises statistic for
        the PIT values by comparing with a uniform distribution between 0 and 1. """
        cvm_stat_and_pval = stats.cramervonmises(self._pit_vals, 'uniform')
        return stat_and_pval(cvm_stat_and_pval.statistic,
                             cvm_stat_and_pval.pvalue)


@PITMetaMetric
class PITAD(PITMeta):
    """ Anderson-Darling statistic """

    def evaluate(self, pit_min=0., pit_max=1.):
        """ Use scipy.stats.anderson_ksamp to compute the Anderson-Darling statistic
        for the PIT values by comparing with a uniform distribution between 0 and 1.
        Up to the current version (1.6.2), scipy.stats.anderson does not support
        uniform distributions as reference for 1-sample test.

        Parameters
        ----------
        pit_min: float, optional
            PIT values below this are discarded
        pit_max: float, optional
            PIT values greater than this are discarded

        Returns
        -------

        """
        pits = self._pit_vals
        mask = (pits >= pit_min) & (pits <= pit_max)
        pits_clean = pits[mask]
        diff = len(pits) - len(pits_clean)
        if diff > 0:
            print(f"{diff} PITs removed from the sample.")
        uniform_yvals = np.linspace(pit_min, pit_max, len(pits_clean))
        ad_results = stats.anderson_ksamp([pits_clean, uniform_yvals])
        stat, crit_vals, sig_lev = ad_results

        return stat_crit_sig(stat, crit_vals, sig_lev)

# comment out for now due to discrete approx
#@PITMetaMetric
#class PITKLD(PITMeta):
#    """ Kullback-Leibler Divergence """
#
#    def __init__(self, pit_vals, pit):
#        super().__init__(pit_vals, pit)
#
#    def evaluate(self, eval_grid=default_quants):
#        """ Use scipy.stats.entropy to compute the Kullback-Leibler
#        Divergence between the empirical PIT distribution and a
#        theoretical uniform distribution between 0 and 1."""
#        warnings.warn("This KLD implementation is based on scipy.stats.entropy, " +
#                      "therefore it uses a discrete distribution of PITs " +
#                      "(internally obtained from PIT object).")
#        pits = self._pit_vals
#        uniform_yvals = np.linspace(0., 1., len(pits))
#        pit_pdf, _ = np.histogram(pits, bins=eval_grid)
#        uniform_pdf, _ = np.histogram(uniform_yvals, bins=eval_grid)
#        kld_metric = stats.entropy(pit_pdf, uniform_pdf)
#        return stat_and_pval(kld_metric, None)
