import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import seaborn as sns



def plot_pdfs(sample, gals, show_ztrue=True, show_photoz_mode=False):
    """Plot a list of individual PDFs using qp plotting function for illustration.
    Ancillary function to be used by class Sample.

    Parameters
    ----------
    gals: `list`
        list of galaxies' indexes
    show_ztrue: `bool`, optional
        if True (default=True), show ztrue as dashed vertical line
    show_photoz_mode: `bool`, optional
        if True (default=False), show photoz_mode as dotted vertical line

    Returns
    -------
    colors: `list`
        list of HTML codes for colors used in the plot lines
    """
    colors = []
    peaks = []
    for i, gal in enumerate(gals):
        peaks.append(sample._pdfs[gal].pdf(sample._photoz_mode[gal]))
        if i == 0:
            axes = sample.pdfs.plot(key=gal, xlim=(0., 2.2), label=f"Galaxy {gal}")
        else:
            _ = sample.pdfs.plot(key=gal, axes=axes, label=f"Galaxy {gal}")
        colors.append(axes.get_lines()[-1].get_color())
        if show_ztrue:
            axes.vlines(sample.ztrue[gal], ymin=0, ymax=100, colors=colors[-1], ls='--')
        if show_photoz_mode:
            axes.vlines(sample.photoz_mode[gal], ymin=0, ymax=100, colors=colors[-1], ls=':')
    plt.ylim(0, np.max(peaks) * 1.05)
    axes.figure.legend()
    return colors


def plot_old_valid(sample, gals=None, colors=None):
    """Plot traditional Zphot X Zspec and N(z) plots for illustration
    Ancillary function to be used by class Sample.

    Parameters
    ----------
    gals: `list`, (optional)
        list of galaxies' indexes
    colors: `list`, (optional)
        list of HTML codes for colors used in the plot highlighted points
    """
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(121)
    plt.plot(sample.ztrue, sample.photoz_mode, 'k,', label=(sample._code).replace("_", " "))
    leg = ax.legend(fancybox=True, handlelength=0, handletextpad=0, loc="upper left")
    for item in leg.legendHandles:
        item.set_visible(False)
    if gals:
        if not colors:
            colors = ['r'] * len(gals)
        for i, gal in enumerate(gals):
            plt.plot(sample.ztrue[gal], sample.photoz_mode[gal], 'o', color=colors[i], label=f'Galaxy {gal}')
    zmax = np.max(sample.ztrue) * 1.05
    plt.xlim(0, zmax)
    plt.ylim(0, zmax)
    plt.xlabel('z$_{true}$')
    plt.ylabel('z$_{phot}$')

    plt.subplot(122)
    sns.kdeplot(sample.ztrue, shade=True, label='z$_{true}$')
    sns.kdeplot(sample.photoz_mode, shade=True, label='z$_{phot}$')
    plt.xlim(0,)
    plt.xlabel('z')
    plt.legend()
    plt.tight_layout()

    sigma_iqr, bias, frac, sigma_mad = old_metrics(sample)
    table = "Metric | Value | DC1 paper  \n :---|---:|---: \n "
    #table += "$$\sigma_{IQR} / (1+z)$$ | "
    table += f" scatter | {sigma_iqr:11.4f} | 0.0154  \n"
    #table += "$$b_{z}$$ | "
    table += f"bias | {bias:11.5f} | -0.00027 \n"
    #table += "$$f_{out}$$ | "
    table += f"outlier rate | {frac:11.3f} | 0.020 "
    return table


def old_metrics(sample):
    point = EvaluatePointStats(sample.photoz_mode, sample.ztrue)
    sigma_iqr = point.CalculateSigmaIQR()
    bias = point.CalculateBias()
    frac = point.CalculateOutlierRate()
    sigma_mad = point.CalculateSigmaMAD()
    return sigma_iqr, bias, frac, sigma_mad


#     z_peak = self._photoz_mode
#     z_weight = None # TBD
# • RMS scatter σ < 0.02(1 + ztrue)
# • bias bz < 0.003
# • catastrophic outlier rate fout < 10 per cent





def plot_pit_qq(metrics, bins=None, title=None, label=None,
                show_pit=True, show_qq=True,
                show_pit_out_rate=True, savefig=False):
    """Quantile-quantile plot
    Ancillary function to be used by class Metrics.

    Parameters
    ----------
    bins: `int`, optional
        number of PIT bins
        if None, use the same number of quantiles (metrics._n_quant)
    title: `str`, optional
        if None, use formatted sample's name (metrics._sample._name)
    label: `str`, optional
        if None, use formatted code's name (metrics._sample._code)
    show_pit: `bool`, optional
        include PIT histogram (default=True)
    show_qq: `bool`, optional
        include QQ plot (default=True)
    show_pit_out_rate: `bool`, optional
        print metric value on the plot panel (default=True)
    savefig: `bool`, optional
        save plot in .png file (default=False)
    """

    if bins is None:
        bins = metrics._n_quant
    if title is None:
        title = (metrics._sample._name).replace("_", " ")
    if label is None:
        label = (metrics._sample._code).replace("_", " ")

        if show_pit_out_rate:
            label += "\n PIT$_{out}$: "
            label += f"{metrics._pit_out_rate:.4f}"

    plt.figure(figsize=[4, 5])
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    if show_qq:
        ax0.plot(metrics.qq_vectors[0], metrics.qq_vectors[1], c='r', linestyle='-',
                 linewidth=3, label=label)
        ax0.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
        ax0.set_ylabel("Qdata", fontsize=18)
        plt.ylim(-0.001, 1.001)
    plt.xlim(-0.001, 1.001)
    plt.title(title)
    if show_pit:
        try:
            y_uni = float(len(metrics._pit)) / float(bins)
        except:
            y_uni = float(len(metrics._pit)) / float(len(bins))
        if not show_qq:
            ax0.hist(metrics._pit, bins=bins, alpha=0.7, label=label)
            ax0.set_ylabel('Number')
            ax0.hlines(y_uni, xmin=0, xmax=1, color='k')
            plt.ylim(0, )  # -0.001, 1.001)
        else:
            ax1 = ax0.twinx()
            ax1.hist(metrics._pit, bins=bins, alpha=0.7)
            ax1.set_ylabel('Number')
            ax1.hlines(y_uni, xmin=0, xmax=1, color='k')

    leg = ax0.legend(handlelength=0, handletextpad=0, fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    if show_qq:
        ax2 = plt.subplot(gs[1])
        ax2.plot(metrics.qq_vectors[0], (metrics.qq_vectors[1] - metrics.qq_vectors[0]), c='r', linestyle='-', linewidth=3)
        plt.ylabel("$\Delta$Q", fontsize=18)
        ax2.plot([0, 1], [0, 0], color='k', linestyle='--', linewidth=2)
        plt.xlim(-0.001, 1.001)
        plt.ylim(np.min([-0.1, np.min(metrics.qq_vectors[1] - metrics.qq_vectors[0])*1.05]),
                 np.max([0.1, np.max(metrics.qq_vectors[1] - metrics.qq_vectors[0])*1.05]))
    if show_pit:
        if show_qq:
            plt.xlabel("Qtheory / PIT Value", fontsize=18)
        else:
            plt.xlabel("PIT Value", fontsize=18)
    else:
        if show_qq:
            plt.xlabel("Qtheory", fontsize=18)

    #plt.tight_layout()
    if savefig:
        fig_filename = str("plot_pit_qq_" +
                           f"{(metrics._sample._code).replace(' ','_')}" +
                           f"_{(metrics._sample._name).replace(' ','_')}.png")
        plt.savefig(fig_filename)
    else:
        fig_filename = None

    return fig_filename

def ks_plot(ks):
    """ KS test illustration.
    Ancillary function to be used by class KS."""

    plt.figure(figsize=[4, 4])
    plt.plot(ks._metrics._xvals, ks._metrics._pit_cdf, 'b-', label="sample PIT")
    plt.plot(ks._metrics._xvals, ks._metrics._uniform_cdf, 'r-', label="uniform")
    ks._bin_stat = np.argmax(np.abs(ks._metrics._pit_cdf - ks._metrics._uniform_cdf))
    plt.vlines(x=ks._metrics._xvals[ks._bin_stat],
               ymin=np.min([ks._metrics._pit_cdf[ks._bin_stat],
                             ks._metrics._uniform_cdf[ks._bin_stat]]),
               ymax=np.max([ks._metrics._pit_cdf[ks._bin_stat],
                            ks._metrics._uniform_cdf[ks._bin_stat]]),
               colors='k')
    plt.plot(ks._metrics._xvals[ks._bin_stat], ks._metrics._pit_cdf[ks._bin_stat], "ko")
    plt.plot(ks._metrics._xvals[ks._bin_stat], ks._metrics._uniform_cdf[ks._bin_stat], "ko")
    plt.xlabel("PIT value")
    plt.ylabel("CDF(PIT)")
    xtext = ks._metrics._xvals[ks._bin_stat]+0.05
    ytext = np.mean([ks._metrics._pit_cdf[ks._bin_stat],
                            ks._metrics._uniform_cdf[ks._bin_stat]])
    plt.text(xtext, ytext, f"KS={ks._stat:.2f}", fontsize=16)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()




class EvaluatePointStats(object):
    """Copied from PZDC1paper repo. Adapted to remove the cut based on magnitude."""

    def __init__(self,pzvec,szvec): #,magvec,imagcut=25.3):
        """An object that takes in the vectors of the point photo-z
        the spec-z, and the i-band magnitudes for calculating the
        point statistics
        Parameters:
        pzvec: Numpy 1d array of the point photo-z values
        szvec: Numpy 1d array of the spec-z values
        magvec: Numpy 1d array of the i-band magnitudes
        imagcut: float: i-band magnitude cut for the sample
        Calculates:
        ez_all: (pz-sz)/(1+sz), the quantity will be useful for calculating statistics
        ez_magcut: ez sample trimmed with imagcut
        """
        self.pzs = pzvec
        self.szs = szvec
        #self.mags = magvec
        #self.imagcut = imagcut
        ez = (pzvec - szvec)/(1.+szvec)
        self.ez_all = ez
        #mask = (magvec<imagcut)
        #ezcut = ez[mask]
        #self.ez_magcut = ezcut

    def CalculateSigmaIQR(self):
        """Calculate the width of the e_z distribution
        using the Interquartile range
        Parameters:
        imagcut: float: i-band magnitude cut for the sample
        Returns:
        sigma_IQR_all float: width of ez distribution for full sample
        sigma_IQR_magcut float: width of ez distribution for magcut sample
        """
        x75,x25 = np.percentile(self.ez_all,[75.,25.])
        iqr_all = x75-x25
        sigma_iqr_all = iqr_all/1.349
        self.sigma_iqr_all = sigma_iqr_all

        # xx75,xx25 = np.percentile(self.ez_magcut,[75.,25.])
        # iqr_cut = xx75 - xx25
        # sigma_iqr_cut= iqr_cut/1.349
        # self.sigma_iqr_magcut = sigma_iqr_cut
        #store the sigmas for the catastrophic outlier calculation

        return sigma_iqr_all#,sigma_iqr_cut

    def CalculateBias(self):
        """calculates the bias of the ez and ez_magcut samples.  In
        keeping with the Science Book, this is just the median of the
        ez values
        Returns:
        bias_all: median of the full ez sample
        bias_magcut: median of the magcut ez sample
        """
        bias_all = np.median(self.ez_all)
        #bias_magcut = np.median(self.ez_magcut)
        return bias_all #,bias_magcut

    def CalculateOutlierRate(self):
        """Calculates the catastrophic outlier rate, defined in the
        Science Book as the number of galaxies with ez larger than
        max(0.06,3sigma).  This keeps the fraction reasonable when
        sigma is very small.
        Returns:
        frac_all: fraction of catastrophic outliers for full sample
        frac_magcut: fraction of catastrophic outliers for magcut
        sample
        """
        num_all = len(self.ez_all)
        #num_magcut = len(self.ez_magcut)
        threesig_all = 3.0*self.sigma_iqr_all
        #threesig_cut = 3.0*self.sigma_iqr_magcut
        cutcriterion_all = np.maximum(0.06,threesig_all)
        #cutcriterion_magcut = np.maximum(0.06,threesig_cut)
        #print("using %.3g for cut for whole sample and %.3g for magcut sample\n"%(cutcriterion_all,cutcriterion_magcut))
        mask_all = (self.ez_all>np.fabs(cutcriterion_all))
        #mask_magcut = (self.ez_magcut>np.fabs(cutcriterion_magcut))
        outlier_all = np.sum(mask_all)
        #outlier_magcut = np.sum(mask_magcut)
        frac_all = float(outlier_all)/float(num_all)
        #frac_magcut = float(outlier_magcut)/float(num_magcut)
        return frac_all #,frac_magcut

    def CalculateSigmaMAD(self):
        """Function to calculate median absolute deviation and sigma
        based on MAD (just scaled up by 1.4826) for the full and
        magnitude trimmed samples of ez values
        Returns:
        sigma_mad_all: sigma_MAD for full sample
        sigma_mad_cut: sigma_MAD for the magnitude cut sample
        """
        tmpmed_all = np.median(self.ez_all)
        #tmpmed_cut = np.median(self.ez_magcut)
        tmpx_all = np.fabs(self.ez_all - tmpmed_all)
        #tmpx_cut = np.fabs(self.ez_magcut - tmpmed_cut)
        mad_all = np.median(tmpx_all)
        #mad_cut = np.median(tmpx_cut)
        sigma_mad_all = mad_all*1.4826
        #sigma_mad_cut = mad_cut*1.4826
        return sigma_mad_all #,sigma_mad_cut
