import qp
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class Metrics():
    pass
#    """ The base class for photo-z PDF quality metrics evaluation.
#        First implementation based on the refactoring of DC1 paper codes:
#        https://github.com/LSSTDESC/PZDC1paper
#        https://arxiv.org/pdf/2001.03621.pdf """
#    def __init__(self):
#        super.__init__()

class Data():

    def __init__(self, pdfs_file, ztrue_file, **kwargs):
        """
        Class to handle input data (pdfs + ztrue)

        Inputs
        ------
        pdfs_file: str
            full path to RAIL's estimation output file (format HDF5)
        ztrue_file: str
            full path to the file containing true redshifts,
            e.g., RAIL's estimation input file (format HDF5)

        Parameters
        ----------
        **kwargs: dict (optional)
            key parameters to read the HDF5 input files, in case
            they are different from RAIL's default output

        Returns
        ------
        object Data
            A Python object to represent a sample of PDFs (format qp.Ensemble)
            and their respective true redshifts.
        """

        self._pdfs_file = pdfs_file
        self._ztrue_file = ztrue_file

        self._pdfs_key = kwargs.get('pdfs_key', "photoz_pdf")
        self._zgrid_key = kwargs.get('zgrid_key', "zgrid")
        self._photoz_mode_key = kwargs.get('photoz_mode', "photoz_mode")
        self._ztrue_key = kwargs.get('ztrue_key', "redshift")

        with h5py.File(self._pdfs_file, 'r') as pf:
            self._pdfs_array = np.array(pf[self._pdfs_key])
            self._zgrid = np.array(pf[self._zgrid_key]).flatten()
            self._photoz_mode = np.array(pf[self._photoz_mode_key])
            self._pdfs = qp.Ensemble(qp.interp, data=dict(xvals=self._zgrid, yvals=self._pdfs_array))

        with h5py.File(self._ztrue_file, 'r') as zf:
            try:
                self._ztrue = np.array(zf['photometry'][self._ztrue_key])
            except:
                try:
                    self._ztrue = np.array(zf[self._ztrue_key])
                except:
                    raise ValueError('Invalid key for true redshift column in ztrue file.')

    @property
    def ztrue(self):
        return self._ztrue

    @property
    def photoz_mode(self):
        return self._photoz_mode

    @property
    def pdfs(self):
        return self._pdfs

    def __len__(self):
        return len(self._ztrue)

    def __str__(self):
        text = str('Dataset:  \n' +
          '========\n'
          f'{len(self._ztrue)} PDFs \n' +
          f'qp representation: {self._pdfs.gen_class.name} \n' +
          f'{len(self._zgrid)} z bins edges from {np.min(self._zgrid)} to {np.max(self._zgrid)}')
        return text

    def plot_pdfs(self, gals):
        colors = []
        for i, gal in enumerate(gals):
            if i == 0:
                axes = self.pdfs.plot(key=gal, xlim=(0., 2.), label=f"Galaxy {gal}")
            else:
                _ = self.pdfs.plot(key=gal, axes=axes, label=f"Galaxy {gal}")
            colors.append(axes.get_lines()[-1].get_color())
            axes.vlines(self.ztrue[gal], ymin=0, ymax=20, colors=colors[-1], ls='--')
        plt.ylim(0, )
        axes.figure.legend()
        return colors

    def old_valid_plots(self, gals, colors=None):
        if not colors:
            colors = ['r']*len(gals)
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(self.ztrue, self.photoz_mode, 'k,')
        for i, gal in enumerate(gals):
            plt.plot(self.ztrue[gal], self.photoz_mode[gal], 'o', color=colors[i], label=f'Galaxy {gal}')
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        plt.ylabel('$z_{true}$')
        plt.xlabel('$z_{phot}$ (mode)')
        plt.subplot(122)
        sns.kdeplot(self.ztrue, shade=True, label='$z_{true}$')
        sns.kdeplot(self.photoz_mode, shade=True, label='$z_{phot}$ (mode)')
        plt.xlabel('z')
        plt.legend()
        plt.tight_layout()


    def PIT(self, using='gridded', dx=0.0001):
        """ computes the Probability Integral Transform (PIT), described in
        Tanaka et al. 2017(ArXiv 1704.05988), which is the integral of the
        PDF from 0 to zref.
        Parameters:
        using: string
             which parameterization to evaluate
        dx: float
             step size used in integral
        Returns
        -------
        ndarray
             The values of the PIT for each ensemble object
             Also stores PIT array in self.pitarray
        """
        if len(self.ztrue) != self.pdfs.npdf:
            print('Warning: number of zref values not equal to number of ensemble objects')
            return
        n = self.pdfs.npdf
        pitlimits = np.zeros([n, 2])
        pitlimits[:, 1] = self.ztrue
        tmppit = self.pdfs.integrate(limits=pitlimits) #, using=using, dx=dx)
        self.pitarray = np.array(tmppit)
        return tmppit



    def QQvectors(self, using, dx=0.0001, Nquants=101):
        """Returns quantile quantile vectors for the ensemble using the PIT values,
        without actually plotting them.  Will be useful in making multi-panel plots
        simply take the percentiles of the values in order to get the Qdata
        quantiles
        Parameters:
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        Nquants: int
            the number of quantile bins to compute, default 100
        Returns
        -------
        numpy arrays for Qtheory and Qdata
        """
        if self.pitarray is not None:
            pits = np.array(self.pitarray)
        else:
            pits = self.PIT(using=using, dx=dx)
            self.pitarray = pits
        quants = np.linspace(0., 100., Nquants)
        Qtheory = quants / 100.
        Qdata = np.percentile(pits, quants)
        return Qtheory, Qdata

    def QQplot(self, using, dx=0.0001, Nquants=101):
        """Quantile quantile plot for the ensemble using the PIT values,
        simply take the percentiles of the values in order to get the Qdata
        quantiles
        Parameters:
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        Nquants: int
            the number of quantile bins to compute, default 100
        Returns
        -------
        matplotlib plot of the quantiles
        """
        if self.pitarray is not None:
            pits = np.array(self.pitarray)
        else:
            pits = self.PIT(using=using, dx=dx)
            self.pitarray = pits
        quants = np.linspace(0., 100., Nquants)
        QTheory = quants / 100.
        Qdata = np.percentile(pits, quants)
        plt.figure(figsize=(10, 10))
        plt.plot(QTheory, Qdata, c='b', linestyle='-', linewidth=3, label='QQ')
        plt.plot([0, 1], [0, 1], color='k', linestyle='-', linewidth=2)
        plt.xlabel("Qtheory", fontsize=18)
        plt.ylabel("Qdata", fontsize=18)
        plt.legend()
        plt.savefig("QQplot.jpg")
        return

    def KS(self, using, dx=0.0001):
        """
        Compute the Kolmogorov-Smirnov statistic and p-value for the PIT
        values by comparing with a uniform distribution between 0 and 1.
        Parameters:
        -----------
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        Returns:
        --------
        KS statistic and pvalue

        """
        if self.pitarray is not None:
            pits = np.array(self.pitarray)
        else:
            pits = np.array(self.PIT(using=using, dx=dx))
            self.pitarray = pits
        ks_result = skgof.ks_test(pits, stats.uniform())
        return ks_result.statistic, ks_result.pvalue

    def CvM(self, using, dx=0.0001):
        """
        Compute the Cramer-von Mises statistic and p-value for the PIT values
        by comparing with a uniform distribution between 0 and 1.
        Parameters:
        -----------
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        Returns:
        --------
        CvM statistic and pvalue

        """
        if self.pitarray is not None:
            pits = np.array(self.pitarray)
        else:
            pits = np.array(self.PIT(using=using, dx=dx))
            self.pitarray = pits
        cvm_result = skgof.cvm_test(pits, stats.uniform())
        return cvm_result.statistic, cvm_result.pvalue

    def AD(self, using, dx=0.0001, vmin=0.005, vmax=0.995):
        """
        Compute the Anderson-Darling statistic and p-value for the PIT
        values by comparing with a uniform distribution between 0 and 1.

        Since the statistic diverges at 0 and 1, PIT values too close to
        0 or 1 are discarded.

        Parameters:
        -----------
        using: string
            which parameterization to evaluate
        dx: float
            step size for integral
        vmin, vmax: floats
            PIT values outside this range are discarded
        Returns:
        --------
        AD statistic and pvalue

        """
        if self.pitarray is not None:
            pits = np.array(self.pitarray)
        else:
            pits = np.array(self.PIT(using=using, dx=dx))
            self.pitarray = pits
        mask = (pits > vmin) & (pits < vmax)
        print("now with proper uniform range")
        delv = vmax - vmin
        ad_result = skgof.ad_test(pits[mask], stats.uniform(loc=vmin, scale=delv))
        return ad_result.statistic, ad_result.pvalue

    def cde_loss(self, grid):
        """Computes the estimated conditional density loss described in
        Izbicki & Lee 2017 (arXiv:1704.08095).

        Parameters:
        grid: np array of values at which to evaluate the pdf.
        Returns:
        an estimate of the cde loss.
        """
        grid, pdfs = self.ensemble_obj.evaluate(grid, norm=True)

        n_obs, n_grid = pdfs.shape

        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(pdfs ** 2, grid))

        # Calculate second term E[f*(Z | X)]
        nns = [np.argmin(np.abs(grid - true_z)) for true_z in self.truths]
        term2 = np.mean(pdfs[range(n_obs), nns])

        return term1 - 2 * term2






""" 
class PIT(Metrics):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class KS(Metrics):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class CvM(Metrics):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class AD(Metrics):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class CRPS(Metrics):
    ''' = continuous rank probability score (Gneiting et al., 2006)'''

    def __init__(self):
        super().__init__()
        raise NotImplementedError

"""
