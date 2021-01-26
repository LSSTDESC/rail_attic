import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_pit_qq(self, bins=None, label=None, title=None,
                show_pit=True, show_qq=True,
                show_pit_out_rate=True, savefig=False):
    """Quantile-quantile plot """
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
        plt.ylim(-0.1, 0.1)
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
        plt.savefig("plot_pit_qq_" +
                    f"{(self._sample._code).replace(' ','_')}" +
                    f"_{(self._sample._name).replace(' ','_')}.png")

