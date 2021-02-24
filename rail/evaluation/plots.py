import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import seaborn as sns


def plot_pdfs(self, gals, show_ztrue=True, show_photoz_mode=False):
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
        peaks.append(self._pdfs[gal].pdf(self._photoz_mode[gal]))
        if i == 0:
            axes = self.pdfs.plot(key=gal, xlim=(0., 2.2), label=f"Galaxy {gal}")
        else:
            _ = self.pdfs.plot(key=gal, axes=axes, label=f"Galaxy {gal}")
        colors.append(axes.get_lines()[-1].get_color())
        if show_ztrue:
            axes.vlines(self.ztrue[gal], ymin=0, ymax=100, colors=colors[-1], ls='--')
        if show_photoz_mode:
            axes.vlines(self.photoz_mode[gal], ymin=0, ymax=100, colors=colors[-1], ls=':')
    plt.ylim(0, np.max(peaks) * 1.05)
    axes.figure.legend()
    return colors


def plot_old_valid(self, gals=None, colors=None):
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
    plt.plot(self.ztrue, self.photoz_mode, 'k,', label=(self._name).replace("_", " "))
    leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    if gals:
        if not colors:
            colors = ['r'] * len(gals)
        for i, gal in enumerate(gals):
            plt.plot(self.ztrue[gal], self.photoz_mode[gal], 'o', color=colors[i], label=f'Galaxy {gal}')
    zmax = np.max(self.ztrue) * 1.05
    plt.xlim(0, zmax)
    plt.ylim(0, zmax)
    plt.ylabel('z$_{true}$')
    plt.xlabel('z$_{phot}$ (mode)')

    plt.subplot(122)
    sns.kdeplot(self.ztrue, shade=True, label='z$_{true}$')
    sns.kdeplot(self.photoz_mode, shade=True, label='z$_{phot}$ (mode)')
    plt.xlabel('z')
    plt.legend()
    plt.tight_layout()


def plot_pit_qq(self, bins=None, title=None, label=None,
                show_pit=True, show_qq=True,
                show_pit_out_rate=True, savefig=False):
    """Quantile-quantile plot
    Ancillary function to be used by class Metrics.

    Parameters
    ----------
    bins: `int`, optional
        number of PIT bins
        if None, use the same number of quantiles (self._n_quant)
    title: `str`, optional
        if None, use formatted sample's name (self._sample._name)
    label: `str`, optional
        if None, use formatted code's name (self._sample._code)
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
        bins = self._n_quant
    if title is None:
        title = (self._sample._name).replace("_", " ")
    if label is None:
        label = (self._sample._code).replace("_", " ")

        if show_pit_out_rate:
            label += "\n PIT$_{out}$: "
            label += f"{self._pit_out_rate:.4f}"

    plt.figure(figsize=[4, 5])
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    if show_qq:
        ax0.plot(self.qq_vectors[0], self.qq_vectors[1], c='r', linestyle='-',
                 linewidth=3, label=label)
        ax0.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
        ax0.set_ylabel("Qdata", fontsize=18)
        plt.ylim(-0.001, 1.001)
    plt.xlim(-0.001, 1.001)
    plt.title(title)
    if show_pit:
        try:
            y_uni = float(len(self._pit)) / float(bins)
        except:
            y_uni = float(len(self._pit)) / float(len(bins))
        if not show_qq:
            ax0.hist(self._pit, bins=bins, alpha=0.7, label=label)
            ax0.set_ylabel('Number')
            ax0.hlines(y_uni, xmin=0, xmax=1, color='k')
            plt.ylim(0, )  # -0.001, 1.001)
        else:
            ax1 = ax0.twinx()
            ax1.hist(self._pit, bins=bins, alpha=0.7)
            ax1.set_ylabel('Number')
            ax1.hlines(y_uni, xmin=0, xmax=1, color='k')

    leg = ax0.legend(handlelength=0, handletextpad=0, fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    if show_qq:
        ax2 = plt.subplot(gs[1])
        ax2.plot(self.qq_vectors[0], (self.qq_vectors[1] - self.qq_vectors[0]), c='r', linestyle='-', linewidth=3)
        plt.ylabel("$\Delta$Q", fontsize=18)
        ax2.plot([0, 1], [0, 0], color='k', linestyle='--', linewidth=2)
        plt.xlim(-0.001, 1.001)
        plt.ylim(np.min([-0.1, np.min(self.qq_vectors[1] - self.qq_vectors[0])*1.05]),
                 np.max([0.1, np.max(self.qq_vectors[1] - self.qq_vectors[0])*1.05]))
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
                           f"{(self._sample._code).replace(' ','_')}" +
                           f"_{(self._sample._name).replace(' ','_')}.png")
        plt.savefig(fig_filename)
    else:
        fig_filename = None

    return fig_filename

def ks_plot(self):
    """ KS test illustration.
    Ancillary function to be used by class KS."""

    plt.figure(figsize=[4, 4])
    plt.plot(self._metrics._xvals, self._metrics._pit_cdf[0], 'b-', label="sample PIT")
    plt.plot(self._metrics._xvals, self._metrics._uniform_cdf[0], 'r-', label="uniform")
    plt.vlines(x=self._metrics._xvals[self._bin_stat],
               ymin=np.min([self._metrics._pit_cdf[0][self._bin_stat],
                             self._metrics._uniform_cdf[0][self._bin_stat]]),
               ymax=np.max([self._metrics._pit_cdf[0][self._bin_stat],
                            self._metrics._uniform_cdf[0][self._bin_stat]]),
               colors='k')
    plt.plot(self._metrics._xvals[self._bin_stat], self._metrics._pit_cdf[0][self._bin_stat], "ko")
    plt.plot(self._metrics._xvals[self._bin_stat], self._metrics._uniform_cdf[0][self._bin_stat], "ko")
    plt.xlabel("PIT value")
    plt.ylabel("CDF(PIT)")
    xtext = self._metrics._xvals[self._bin_stat]+0.05
    ytext = np.mean([self._metrics._pit_cdf[0][self._bin_stat],
                            self._metrics._uniform_cdf[0][self._bin_stat]])
    plt.text(xtext, ytext, f"KS={self._stat:.2f}", fontsize=16)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
