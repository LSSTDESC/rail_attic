import qp
import h5py
import numpy as np
import plots

class Sample:
    """
    Handle photo-z output data (pdfs + ztrue) of
    a given sample. Inherits from qp.Ensemble."""

    def __init__(self, pdfs_file, ztrue_file, code="", name="", **kwargs):
        """Class constructor

        Parameters
        ----------
        pdfs_file: `str`
            full path to RAIL's estimation output file (format HDF5)
        ztrue_file: `str`
            full path to the file containing true redshifts,
            e.g., RAIL's estimation input file (format HDF5)
        code: `str`, (optional)
            algorithm name (for plot legends)
        name: `str`, (optional)
            sample name (for plot legends)
        **kwargs: `dict`, (optional)
            key parameters to read the HDF5 input files, in case
            they are different from RAIL's default output
        """
        self._pdfs_file = pdfs_file
        self._ztrue_file = ztrue_file
        self._code = code
        self._name = name

        self._pdfs_key = kwargs.get('pdfs_key', "photoz_pdf")
        self._zgrid_key = kwargs.get('zgrid_key', "zgrid")
        self._photoz_mode_key = kwargs.get('photoz_mode', "photoz_mode")
        self._ztrue_key = kwargs.get('ztrue_key', "redshift")

        pdfs_file_format = (self._pdfs_file.split(".")[-1]).lower()

        if pdfs_file_format == "out":
            print("Validation file from DC1 paper!")
            self._ztrue = np.loadtxt(self._ztrue_file, unpack=True, usecols=[2])
            self._pdfs_array = np.loadtxt(self._pdfs_file)
            path = "/".join(self._pdfs_file.split("/")[:-1])
            self._zgrid  = np.loadtxt(path + "/zarrayfile.out")
            self._photoz_mode = np.array([self._zgrid[np.argmax(pdf)] for pdf in self._pdfs_array])
        elif pdfs_file_format == "hdf5":
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
        elif pdfs_file_format == "pz":
            print("DNF format")
            self._photoz_mode, self._ztrue = np.loadtxt(self._ztrue_file, unpack=True, usecols=[0,2])
            pdfs_file_array = np.loadtxt(self._pdfs_file)
            self._pdfs_array = pdfs_file_array[1:, 3:]
            self._zgrid = pdfs_file_array[0, 3:]
        else:
            raise ValueError(f"PDFs input file format {pdfs_file_format} is not supported.")

        self._pdfs = qp.Ensemble(qp.interp, data=dict(xvals=self._zgrid,
                                                      yvals=self._pdfs_array))

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
    def zgrid(self):
        """Redshift grid (binning)"""
        return self._zgrid


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
          f'{len(self)} PDFs with {len(self._pdfs_array[0])} probabilities each \n' +
          f'qp representation: {self._pdfs.gen_class.name} \n' +
          f'z grid: {len(self._zgrid)} z values from {np.min(self._zgrid)} to {np.max(self._zgrid)} inclusive')

        return text

    def plot_pdfs(self, gals, show_ztrue=True, show_photoz_mode=False):
        colors = plots.plot_pdfs(self, gals, show_ztrue=show_ztrue,
                                 show_photoz_mode=show_photoz_mode)
        return colors

    def plot_old_valid(self, gals=None, colors=None):
        plots.plot_old_valid(self, gals=gals, colors=colors)



