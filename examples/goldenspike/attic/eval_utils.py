import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from rail.evaluation.metrics.pit import *
from qp.ensemble import Ensemble
from qp import interp


def plot_pit_qq(pdf_ens, ztrue, qbins=101, title=None, code=None,
                show_pit=True, show_qq=True,
                pit_out_rate=None, outdir="", savefig=False) -> str:
    """Quantile-quantile plot
    Ancillary function to be used by class Metrics.

    Parameters
    ----------
    pit: `PIT` object
        class from metrics.py
    qbins: `int`, optional
        number of PIT bins/quantiles
    title: `str`, optional
        if None, use formatted sample's name (sample.name)
    label: `str`, optional
        if None, use formatted code's name (sample.code)
    show_pit: `bool`, optional
        include PIT histogram (default=True)
    show_qq: `bool`, optional
        include QQ plot (default=True)
    pit_out_rate: `ndarray`, optional
        print metric value on the plot panel (default=None)
    savefig: `bool`, optional
        save plot in .png file (default=False)
    """

    if qbins is None:
        qbins = 100
    if title is None:
        title = ""

    if code is None:
        code = ""
        label = ""
    else:
        label = code + "\n"


    if pit_out_rate is not None:
        try:
            label += "PIT$_{out}$: "
            label += f"{float(pit_out_rate):.4f}"
        except:
            print("Unsupported format for pit_out_rate.")

    plt.figure(figsize=[4, 5])
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])

    pitobj = PIT(pdf_ens, ztrue)
    spl_ens, metamets = pitobj.evaluate()
    pit_vals = np.array(pitobj._pit_samps)
    pit_out_rate = PITOutRate(pit_vals, spl_ens).evaluate()

    # #################
    q_theory = np.linspace(0., 1., qbins)
    q_data = np.quantile(pit_vals, q_theory)
    qq_vec = (q_theory, q_data)
    
    if show_qq:
        ax0.plot(qq_vec[0], qq_vec[1], c='r',
                 linestyle='-', linewidth=3)#, label=label)
        ax0.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
        ax0.set_ylabel("Q$_{data}$", fontsize=18)
        plt.ylim(-0.001, 1.001)
    plt.xlim(-0.001, 1.001)
    plt.title(title)
    if show_pit:
        try:
            y_uni = float(len(pit_vals)) / float(qbins)
        except:
            y_uni = float(len(pit_vals)) / float(len(qbins))
        if not show_qq:
            ax0.hist(pit_vals, bins=qbins, alpha=0.7) #, label=label)
            ax0.set_ylabel('Number')
            ax0.hlines(y_uni, xmin=0, xmax=1, color='k')
            plt.ylim(0, )  # -0.001, 1.001)
        else:
            ax1 = ax0.twinx()
            ax1.hist(pit_vals, bins=qbins, alpha=0.7)
            ax1.set_ylabel('Number')
            ax1.hlines(y_uni, xmin=0, xmax=1, color='k')
    leg = ax0.legend(handlelength=0, handletextpad=0, fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    if show_qq:
        ax2 = plt.subplot(gs[1])
        ax2.plot(qq_vec[0], (qq_vec[1] - qq_vec[0]), c='r', linestyle='-', linewidth=3)
        plt.ylabel("$\Delta$Q", fontsize=18)
        ax2.plot([0, 1], [0, 0], color='k', linestyle='--', linewidth=2)
        plt.xlim(-0.001, 1.001)
        plt.ylim(np.min([-0.12, np.min(qq_vec[1] - qq_vec[0]) * 1.05]),
                 np.max([0.12, np.max(qq_vec[1] - qq_vec[0]) * 1.05]))
    if show_pit:
        if show_qq:
            plt.xlabel("Q$_{theory}$ / PIT Value", fontsize=18)
        else:
            plt.xlabel("PIT Value", fontsize=18)
    else:
        if show_qq:
            plt.xlabel("Q$_{theory}$", fontsize=18)
    if savefig:
        fig_filename = str("plot_pit_qq_" +
                           f"{(code).replace(' ', '_')}.jpg")
        plt.savefig(f"{outdir}/{fig_filename}", format='jpg')
    else:
        fig_filename = None

    return fig_filename


def ks_plot(pitobj, n_quant=100, savefig=True, figname='default_ksplot.jpg'):
    """ KS test illustration.
    Ancillary function to be used by class KS."""
    #pits = ks._pits
    spl_ens, metamets = pitobj.evaluate()
    pits = np.array(pitobj._pit_samps)
    ksobj = PITKS(pits, spl_ens)
    stat_and_pval = ksobj.evaluate()
    xvals = np.linspace(0., 1., n_quant)
    yvals = np.array([np.histogram(pits, bins=len(xvals))[0]])
    pit_cdf = Ensemble(interp, data=dict(xvals=xvals, yvals=yvals)).cdf(xvals)[0]
    uniform_yvals = np.array([np.full(n_quant, 1.0 / float(n_quant))])
    uniform_cdf = Ensemble(interp, data=dict(xvals=xvals, yvals=uniform_yvals)).cdf(xvals)[0]

    plt.figure(figsize=[4, 4])
    plt.plot(xvals, uniform_cdf, 'r-', label="uniform")
    plt.plot(xvals, pit_cdf, 'b-', label="sample PIT")
    bin_stat = np.argmax(np.abs(pit_cdf - uniform_cdf))

    plt.vlines(x=xvals[bin_stat],
               ymin=np.min([pit_cdf[bin_stat], uniform_cdf[bin_stat]]),
               ymax=np.max([pit_cdf[bin_stat], uniform_cdf[bin_stat]]),
               colors='k')
    plt.plot(xvals[bin_stat], pit_cdf[bin_stat], "k.")
    plt.plot(xvals[bin_stat], uniform_cdf[bin_stat], "k.")
    ymean = (pit_cdf[bin_stat] + uniform_cdf[bin_stat]) / 2.
    plt.text(xvals[bin_stat] + 0.05, ymean, "max", fontsize=16)
    plt.xlabel("PIT value")
    plt.ylabel("CDF(PIT)")
    xtext = 0.63
    ytext = 0.03
    plt.text(xtext, ytext, f"KS={stat_and_pval.statistic:.4f}", fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format='jpg')

def plot_point_est(zpoint, z_true, sigma, code, outfile):
    plt.figure()
    plt.scatter(z_true, zpoint, s=1, c='k', alpha=0.7)
    plt.xlabel("redshift", fontsize=18)
    plt.ylabel("point photo-z", fontsize=18)
    plt.plot([0, 3], [0, 3], 'k--', lw=1)
    upper_l = sigma
    upper_h = 3.0 + 3.0*(sigma)
    lower_l = -1.*sigma
    lower_h = 3.0 - 3.0*(sigma)
    plt.plot([0., 3.],[upper_l, upper_h], c='r', lw=1, linestyle='--')
    plt.plot([0., 3.],[lower_l, lower_h], c='r', lw=1, linestyle='--')
    #plt.title(f"point estimate for {code}", fontsize=18)
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.savefig(outfile, format='jpg')
