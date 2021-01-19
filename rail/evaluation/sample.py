import qp
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Sample:
    """
    Handle photo-z output data (pdfs + ztrue) of
    a given sample. Inherits from qp.Ensemble.

    Parameters
    ----------
    pdfs_file: `str`
        full path to RAIL's estimation output file (format HDF5)
    ztrue_file: `str`
        full path to the file containing true redshifts,
        e.g., RAIL's estimation input file (format HDF5)
    name: `str`, (optional)
        sample name (for plot legends)
    **kwargs: `dict`, (optional)
        key parameters to read the HDF5 input files, in case
        they are different from RAIL's default output
    """

    def __init__(self, pdfs_file, ztrue_file, code="", name="", **kwargs):

        self._pdfs_file = pdfs_file
        self._ztrue_file = ztrue_file
        self._code = code
        self._name = name

        self._pdfs_key = kwargs.get('pdfs_key', "photoz_pdf")
        self._zgrid_key = kwargs.get('zgrid_key', "zgrid")
        self._photoz_mode_key = kwargs.get('photoz_mode', "photoz_mode")
        self._ztrue_key = kwargs.get('ztrue_key', "redshift")

        with h5py.File(self._ztrue_file, 'r') as zf:
            try:
                self._ztrue = np.array(zf['photometry'][self._ztrue_key])
            except:
                try:
                    self._ztrue = np.array(zf[self._ztrue_key])
                except:
                    raise ValueError('Invalid key for true redshift column in ztrue file.')


        with h5py.File(self._pdfs_file, 'r') as pf:
            self._pdfs_array = np.array(pf[self._pdfs_key])
            self._zgrid = np.array(pf[self._zgrid_key]).flatten()
            self._photoz_mode = np.array(pf[self._photoz_mode_key])
            self._pdfs = qp.Ensemble(qp.interp, data=dict(xvals=self._zgrid, yvals=self._pdfs_array))

    @property
    def code(self):
        """Photo-z code/algorithm name"""
        return self._code
    
    @property
    def name(self):
        """Sample name"""
        return self._name

    @property
    def ztrue(self):
        """True redshifts array"""
        return self._ztrue

    @property
    def photoz_mode(self):
        """Photo-z (mode) array"""
        return self._photoz_mode

    @property
    def pdfs(self):
        """qp.Ensemble object containing the PDFs ('interp' representation)"""
        return self._pdfs

    def __len__(self):
        if (len(self._ztrue) != (self._pdfs.npdf)):
            raise ValueError("Number of pdfs and true redshifts do not match!!!")
        return len(self._ztrue)

    def __str__(self):
        code_str = f'Algorithm: {self._code}'
        name_str = f'Sample: {self._name}'
        line_str = '-' * (max(len(code_str), len(name_str)))
        text = str(line_str + '\n' +
          name_str +'\n' +
          code_str + '\n' +
          line_str + '\n' +
          f'{len(self)} PDFs \n' +
          f'qp representation: {self._pdfs.gen_class.name} \n' +
          f'{len(self._zgrid)} z bins edges from {np.min(self._zgrid)} to {np.max(self._zgrid)}')
          
        return text

    def plot_pdfs(self, gals):
        """Plot a list of individual PDFs using qp plotting function for illustration.

        Parameters
        ----------
        gals: `list`
            list of galaxies' indexes

        Returns
        -------
        colors: `list`
            list of HTML codes for colors used in the plot lines
        """
        colors = []
        for i, gal in enumerate(gals):
            if i == 0:
                axes = self.pdfs.plot(key=gal, xlim=(0., 2.2), label=f"Galaxy {gal}")
            else:
                _ = self.pdfs.plot(key=gal, axes=axes, label=f"Galaxy {gal}")
            colors.append(axes.get_lines()[-1].get_color())
            axes.vlines(self.ztrue[gal], ymin=0, ymax=20, colors=colors[-1], ls='--')
        plt.ylim(0, 15)
        axes.figure.legend()
        return colors

    def plot_old_valid(self, gals=None, colors=None):
        """Plot traditional Zphot X Zspec and N(z) plots for illustration

        Parameters
        ----------
        gals: `list`
            list of galaxies' indexes
        """
        plt.figure(figsize=(10, 4))
        ax = plt.subplot(121)
        plt.plot(self.ztrue, self.photoz_mode, 'k,', label=self._name)
        leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True)
        for item in leg.legendHandles:
            item.set_visible(False)
        if gals:
            if not colors:
                colors = ['r'] * len(gals)
            for i, gal in enumerate(gals):
                plt.plot(self.ztrue[gal], self.photoz_mode[gal], 'o', color=colors[i], label=f'Galaxy {gal}')
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        plt.ylabel('z$_{true}$')
        plt.xlabel('z$_{phot}$ (mode)')




        plt.subplot(122)
        sns.kdeplot(self.ztrue, shade=True, label='z$_{true}$')
        sns.kdeplot(self.photoz_mode, shade=True, label='z$_{phot}$ (mode)')
        plt.xlabel('z')
        plt.legend()
        plt.tight_layout()
