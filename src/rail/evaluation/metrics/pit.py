import inspect
import numpy as np
from scipy import stats
import qp
from .base import MetricEvaluator
from rail.evaluation.utils import stat_and_pval, stat_crit_sig
from sklearn.preprocessing import StandardScaler
from src.rail.evaluation.metrics.condition_pit_utils.mlp_training import train_local_pit, load_model, get_local_pit, trapz_grid
from joblib import Parallel, delayed
from src.rail.evaluation.metrics.condition_pit_utils.ispline import fit_cdf
from src.rail.evaluation.metrics.condition_pit_utils.utils import get_pit
from tqdm import trange
import matplotlib.pyplot as plt


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


class UnconditionPIT(MetricEvaluator):
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


class ConditionPIT(MetricEvaluator):
    #def __init__(self, qp_ens_cde_calib, qp_ens_cde_test, z_grid, z_true_calib, z_true_test,
    #             features_calib, features_test):
    def __init__(self, cde_calib, cde_test, z_grid, z_true_calib, z_true_test,
                 features_calib, features_test, qp_ens_cde_calib):
        super().__init__(qp_ens_cde_calib)

        # cde conditional density estimate
        # reading and storing
        # input are PDFs evaluated with a photo=z method on representative sample of objects,
        # features and ztrue (zspec for real data).
        # calculate uncondition pit on the fly because it's input to training
        # up to block 10 in bitrateep notebook goes outside, but the standard scaling will be done inside init
        # because the scaler itself is needed for both training and evaluation
        # pit_calib = get_pit(cde_calib, z_grid, z_calib)
        # pit_test = get_pit(cde_test, z_grid, z_test)
        # the train are going to be those galaxies that have spectroscopic redshifts, test the others

        # single leading underscore is indicates to the user of the class that the attribute should only be accessed
        # by the class's internals (or perhaps those of a subclass) and that they need not directly access it
        # and probably shouldn't modify it. when you import everything from the
        # class you don't import objects whose name starts with an underscore
        #self._qp_ens_cde_test = qp_ens_cde_test
        self._cde_calib = cde_calib
        self._cde_test = cde_test
        self._zgrid = z_grid
        self._ztrue_calib = z_true_calib
        self._ztrue_test = z_true_test
        self._features_calib = features_calib
        self._features_test = features_test

        # now let's apply the standard scaler
        scaler = StandardScaler()
        self.x_calib = scaler.fit_transform(self._features_calib) # with or without the underscore?
        self.x_test = scaler.transform(self._features_test)

        # now let's do pit using Bitrateep utils get_pit
        self.uncond_pit_calib = get_pit(cde_calib, z_grid, self._ztrue_calib)
        self.uncond_pit_test = get_pit(cde_test, z_grid, self._ztrue_test)

        # now let's do pit using the unconditional pit coded above
        # uncond_pit_calib_class = UnconditionPIT(self._qp_ens, self._ztrue_calib)
        # self.uncond_pit_calib = uncond_pit_calib_class.evaluate(eval_grid=self._zgrid)
        # uncond_pit_test_class = UnconditionPIT(self._qp_ens_cde_test, self._ztrue_test)
        # self.uncond_pit_test = uncond_pit_test_class.evaluate(eval_grid=self._zgrid)

    def train(self, patience=10, n_epochs=10000, lr=0.001, weight_decay=0.01, batch_size=2048, frac_mlp_train=0.9,
              lr_decay=0.95, oversample=50, n_alpha=201, checkpt_path="./checkpoint_GPZ_wide_CDE_1024x512x512.pt",
              hidden_layers=None):
        # training, hyperparameters need to be tuned
        if hidden_layers is None:
            hidden_layers = [256, 256, 256]
        rhat, _, _ = train_local_pit(X=self.x_calib, pit_values=self.uncond_pit_calib, patience=patience,
                                     n_epochs=n_epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                                     frac_mlp_train=frac_mlp_train, lr_decay=lr_decay, trace_func=print,
                                     oversample=oversample, n_alpha=n_alpha, checkpt_path=checkpt_path,
                                     hidden_layers=hidden_layers)

    def evaluate(self, eval_grid=default_quants, meta_options=None, model_checkpt_path='model_checkpt_path',
                 model_hidden_layers=None, nn_type='monotonic', batch_size=100, num_basis=40,
                 num_cores=1):
        # we just need the features X test since the model has been trained in the function train and we just need to
        # run the model on the features to obtain directly the calibrated PDFs.
        # get pit local and ispline fits

        if meta_options is None:
            meta_options = _pitMetaMetrics
        if model_hidden_layers is None:
            model_hidden_layers = [1024, 512, 512]

        rhat = load_model(input_size=self.x_test.shape[1] + 1, hidden_layers=model_hidden_layers,
                          checkpt_path=model_checkpt_path, nn_type=nn_type)
        self.alphas = np.linspace(0.0, 1, len(self._zgrid))
        pit_local = get_local_pit(rhat, self.x_test, alphas=self.alphas, batch_size=batch_size)

        self.cdf_test = trapz_grid(self._cde_test, self._zgrid)
        self.cdf_test[self.cdf_test > 1] = 1

        pit_local_fit, _, _ = zip(*Parallel(n_jobs=num_cores)(
            delayed(fit_cdf)(self.alphas, pit_local[i, :], self.cdf_test[i, :], num_basis=num_basis) for i in
            trange(len(pit_local))))

        return pit_local, np.array(pit_local_fit)

    def diagnostics(self, pit_local, pit_local_fit):
        # P-P plot creation, not one for every galaxy but something clever
        rng = np.random.default_rng(42)
        random_idx = rng.choice(len(self.x_test), 25, replace=False)
        fig, axs = plt.subplots(5,5, figsize=(15, 15))
        axs = np.ravel(axs)

        for count, index in enumerate(random_idx):
            axs[count].scatter(self.alphas, pit_local[index], s=1)
            axs[count].scatter(self.cdf_test[index], pit_local_fit[index], c="C1")
            axs[count].plot(self._zgrid, self.cdf_test[index], c="k")
            axs[count].plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), color="k", ls="--")
            axs[count].set_xlim(0, 1)
            axs[count].set_ylim(0, 1)
            axs[count].set_aspect("equal")
        fig.suptitle("Local P-P plot", fontsize=30)

        fig.text(0.5,-0.05,"Theoretical P", fontsize=30)
        fig.text(-0.05,0.5,"Empirical P", rotation=90, fontsize=30)
        plt.tight_layout()
        plt.show()

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
