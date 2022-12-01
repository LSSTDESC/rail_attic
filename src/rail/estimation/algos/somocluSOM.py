import numpy as np
#from minisom import MiniSom
from somoclu import Somoclu
from pathos.pools import ProcessPool
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatInformer
from rail.estimation.summarizer import SZPZSummarizer
from rail.core.data import QPHandle, TableHandle
import qp

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def_bands = ['u', 'g', 'r', 'i', 'z', 'y']
def_cols = [f"mag_{band}_lsst" for band in def_bands]
def_maglims = dict(mag_u_lsst=27.79,
                   mag_g_lsst=29.04,
                   mag_r_lsst=29.06,
                   mag_i_lsst=28.62,
                   mag_z_lsst=27.98,
                   mag_y_lsst=27.05)


def _computemagcolordata(data, ref_column_name, column_names, colusage):
    numcols = len(column_names)
    if colusage == 'magandcolors':
        coldata = np.array(data[ref_column_name])
        for i in range(numcols - 1):
            tmpcolor = data[column_names[i]] - data[column_names[i + 1]]
            coldata = np.vstack((coldata, tmpcolor))
    elif colusage == 'colors':
        coldata = np.array(data[column_names[0]] - data[column_names[1]])
        for i in range(numcols - 2):
            tmpcolor = data[column_names[i + 1]] - data[column_names[i + 2]]
            coldata = np.vstack((coldata, tmpcolor))
    elif colusage == 'columns':  # pragma: no cover
        coldata = np.array(data[column_names[0]])
        for i in range(numcols - 1):
            coldata = np.vstack((coldata, np.array(data[column_names[i + 1]])))
    else: # pragma: no cover
        raise ValueError(f"column usage value {colusage} is not valid, valid values are 'colors', 'magandcolors', and 'columns'")
    return coldata.T


def get_bmus(som, data, step=1000):  # pragma: no cover
    '''
    This function gets the "best matching unit (bmu)" of a given data on a pre-trained SOM.
    It works by multiprocessing chunks of the data.
    Input:
    som: a pre-trained Somoclu object;
    data: np.ndarray of the data vector;
    step: int, the size of a chunk of the data.
    '''

    def func(i):
        if i * step + step > len(data):
            dmap = som.get_surface_state(data[i*step:])
            bmus = np.zeros((step, 2))
            bmus[:len(data)-i*step] = som.get_bmus(dmap).tolist()
            return bmus
        else:
            dmap = som.get_surface_state(data[i*step:i*step+step])
            return som.get_bmus(dmap).tolist()
    n_chunk = int(np.ceil(len(data) / step))
    with ProcessPool() as p:
        bmus = p.map(func, np.arange(n_chunk))
    bmus_array = np.asarray(bmus).astype(np.int)
    return bmus_array.reshape(bmus_array.shape[0]*bmus_array.shape[1], 2)[:len(data)]


def plot_som(ax, som_map, grid_type='rectangular', colormap=cm.viridis, cbar_name=None,
             vmin=None, vmax=None):  # pragma: no cover
    '''
    This function plots the pre-trained SOM.
    Input:
    ax: the axis to be plotted on.
    som_map: a 2-D array contains the value in a pre-trained SOM. The value can be the number 
    of sources in each cell; or the mean feature in every cell.  
    grid_type: string, either 'rectangular' or 'hexagonal'.
    colormap: the colormap to show the values. default: cm.viridis.
    cbar_name: the label on the color bar.
    '''
    if vmin == None and vmax == None:
        vmin = np.quantile(som_map[~np.isnan(som_map)],0.01)
        vmax = np.quantile(som_map[~np.isnan(som_map)],0.99)
    cscale = (som_map-vmin) / (vmax - vmin)
    som_dim = cscale.shape[0]
    if grid_type == 'rectangular':
        ax.matshow(som_map.T, cmap=colormap, 
                   vmin=vmin, 
                   vmax=vmax)
    else:
        yy, xx= np.meshgrid(np.arange(som_dim), np.arange(som_dim))
        shift = np.zeros(som_dim)
        shift[::2]=-0.5
        xx = xx + shift
        for i in range(cscale.shape[0]):
            for j in range(cscale.shape[1]):
                wy = yy[(i, j)] * np.sqrt(3) / 2
                if np.isnan(cscale[i,j]):
                    color = 'k'
                else:
                    color = colormap(cscale[i,j])

                hex = RegularPolygon((xx[(i, j)], wy), 
                                 numVertices=6, 
                                 radius= 1 / np.sqrt(3),
                                 facecolor=color, 
                                 edgecolor=color,
                                 #alpha=.4, 
                                 lw=0.2,)
                ax.add_patch(hex)

    scmap = plt.scatter([0,0],[0,0], s=0, c=[vmin, vmax], 
                            cmap=colormap)
    ax.set_xlim(-1,som_dim-.5)
    ax.set_ylim(-0.5,som_dim * np.sqrt(3) / 2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cb = plt.colorbar(scmap, cax=cax)
    cb.ax.tick_params(labelsize=15)
    cb.set_label(cbar_name, size=15)
    ax.axis('off')


class Inform_somocluSOMSummarizer(CatInformer):
    """Summarizer that uses a SOM to construct a weighted sum
    of spec-z objects in the same SOM cell as each photometric
    galaxy in order to estimate the overall N(z).  This is
    very related to the NZDir estimator, though that estimator
    actually reverses this process and looks for photometric
    neighbors around each spectroscopic galaxy, which can
    lead to problems if there are photometric galaxies with
    no nearby spec-z objects (NZDir is not aware that such
    objects exist and thus can hid biases).

    We apply somoclu package (https://somoclu.readthedocs.io/)
    to train the SOM.

    Part of the SOM estimator will be a check for cells
    which contain photometric objects but do not contain any
    corresponding training/spec-z objects, those unmatched
    objects will be flagged for possible removal from the
    input sample.
    The inform stage will simply construct a 2D grid SOM
    using somoclu from a large sample of input
    photometric data and save this as an output.  This may
    be a computationally intensive stage, though it will
    hopefully be run once and used by the estimate/summarize
    stage many times without needing to be re-run.

    We can make the SOM either with all colors, or one
    magnitude and N colors, or an arbitrary set of columns.
    The code includes a flag `column_usage` to set usage,
    If set to "colors" it will take the difference of each
    adjacen pair of columns in `usecols` as the colors. If
    set to `magandcolors` it will use these colors plus one
    magnitude as specified by `ref_column_name`.  If set to
    `columns` then it will take as inputs all of the columns
    specified by `usecols` (they can be magnitudes, colors,
    or any other input specified by the user).  NOTE: any
    custom `usecols` parameters must have an accompanying
    `nondetect_val` dictionary that will replace
    nondetections with the nondetect_val values!

    Returns
    -------
    model: pickle file
      pickle file containing the `somoclu` SOM object that
    will be used by the estimation/summarization stage
    """
    name = 'Inform_SOMoclu'
    config_options = CatInformer.config_options.copy()
    config_options.update(usecols=Param(list, def_cols, msg="columns used to construct SOM"),
                          column_usage=Param(str, "magandcolors", msg="switch for how SOM uses columns, valid values are 'colors', 'magandcolors', and 'columns'"),
                          ref_column_name=Param(str, 'mag_i_lsst', msg="name for mag column used if column_usage is set to 'magsandcolors'"),
                          nondetect_val=Param(float, 99.0, msg="value to be replaced with magnitude limit for non detects"),
                          mag_limits=Param(dict, def_maglims, msg="1 sigma mag limits"),
                          seed=Param(int, 0, msg="Random number seed"),
                          n_rows=Param(int, 31, msg="number of cells in SOM y dimension"),
                          n_columns=Param(int, 31, msg="number of cells in SOM x dimension"),
                          gridtype=Param(str, 'rectangular', msg="Optional parameter to specify the grid form of the nodes:"
                                         + "* 'rectangular': rectangular neurons (default)"
                                         + "* 'hexagonal': hexagonal neurons"),
                          maptype=Param(str, 'planar', msg="Optional parameter to specify the map topology:"
                                        + "* 'planar': Planar map (default)"
                                        + "* 'toroid': Toroid map"),
                          std_coeff=Param(float, 1.5, msg="Optional parameter to set the coefficient in the Gaussian"
                                          + "neighborhood function exp(-||x-y||^2/(2*(coeff*radius)^2))"
                                          + "Default: 1.5"),
                          som_learning_rate=Param(float, 0.5, msg="Initial SOM learning rate (scale0 param in Somoclu)"),
                          hdf5_groupname=Param(str, "photometry", msg="name of hdf5 group for data, if None, then set to ''"))

    def __init__(self, args, comm=None):
        """ Constructor:
        Do Informer specific initialization """
        CatInformer.__init__(self, args, comm=comm)
        self.model = None

    def run(self):
        """Build a SOM from photometric data
        **NOT** spectroscopic data!
        """
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data('input')
        # replace nondetects
        for col in self.config.usecols:
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                mask = np.isnan(training_data[col])
            else:
                mask = np.isclose(training_data[col], self.config.nondetect_val)
            training_data[col][mask] = self.config.mag_limits[col]

        colors = _computemagcolordata(training_data, self.config.ref_column_name,
                                      self.config.usecols, self.config.column_usage)

        som = Somoclu(self.config.n_columns, self.config.n_rows,
                      gridtype=self.config.gridtype,
                      maptype=self.config.maptype, initialization='pca')

        som.train(colors)

        modeldict = dict(som=som, usecols=self.config.usecols,
                         ref_column=self.config.ref_column_name,
                         n_rows=self.config.n_rows, n_columns=self.config.n_columns,
                         column_usage=self.config.column_usage)
        self.model = modeldict
        self.add_data('model', self.model)


class somocluSOMSummarizer(SZPZSummarizer):
    """Quick implementation of a SOM-based summarizer that
    constructs and N(z) estimate via a weighted sum of the
    empirical N(z) consisting of the normalized histogram
    of spec-z values contained in the same SOM cell as
    each photometric galaxy.
    There are some general guidelines to choosing the geometry
    and number of total cells in the SOM.  This paper:
    http://www.giscience2010.org/pdfs/paper_230.pdf
    recommends 5*sqrt(num rows * num data columns) as a rough
    guideline.  Some authors state that a SOM with one
    dimension roughly twice as long as the other are better,
    while others find that square SOMs with equal X and Y
    dimensions are best, the user can set the dimensions
    using the n_columns and n_rows parameters.
    For more discussion on SOMs and photo-z calibration, see
    the KiDS paper on the topic:
    http://arxiv.org/abs/1909.09632
    particularly the appendices.
    Note that several parameters are stored in the model file,
    e.g. the columns used. This ensures that the same columns
    used in constructing the SOM are used when finding the
    winning SOM cell with the test data.
    Two additional files are also written out:
    `cellid_output` outputs the 'winning' SOM cell for each
    photometric galaxy, in both raveled and 2D SOM cell
    coordinates.  If the objectID or galaxy_id is present
    they will also be included in this file, if not the
    coordinates will be written in the same order in which
    the data is read in.
    `uncovered_cell_file` outputs the raveled cell
    IDs of cells that contain photometric
    galaxies but no corresponding spectroscopic objects,
    these objects should be removed from the sample as they
    cannot be accounted for properly in the summarizer.
    Some iteration on data cuts may be necessary to
    remove/mitigate these 'uncovered' objects.

    Parameters
    ----------
    zmin: float
      min redshift for z grid
    zmax: float
      max redshift for z grid
    nzbins: int
      number of bins in z grid
    hdf5_groupname: str
      hdf5 group name for photometric data, set to "" if data is at top leve of hdf5 file
    spec_groupname: str
      hdf5 group name for spectroscopic data, set to "" if data is at top leve of hdf5 file
    phot_weightcol: str
      name of photometric weight column.  If no weights are to be used, set to ''
    spec_weightcol: str
      column name of the spectroscopic weight column.  If no weights are to be used, set to ''
    nsamples: int
      number of bootstrap spec-z samples to generate

    Returns
    -------
    qp_ens: qp Ensemble
      ensemble of bootstrap realizations of the estimated N(z) for the input photometric data
    """
    name = 'somocluSOMSummarizer'
    config_options = SZPZSummarizer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          hdf5_groupname=Param(str, "photometry", msg="name of hdf5 group for data, if None, then set to ''"),
                          objid_name=Param(str, "", "name of ID column, if present will be written to cellid_output"),
                          nondetect_val=Param(float, 99.0, msg="value to be replaced with magnitude limit for non detects"),
                          mag_limits=Param(dict, def_maglims, msg="1 sigma mag limits"),
                          spec_groupname=Param(str, "photometry", msg="name of hdf5 group for spec data, if None, then set to ''"),
                          seed=Param(int, 12345, msg="random seed"),
                          redshift_colname=Param(str, "redshift", msg="name of redshift column in specz file"),
                          phot_weightcol=Param(str, "", msg="name of photometry weight, if present"),
                          spec_weightcol=Param(str, "", msg="name of specz weight col, if present"),
                          nsamples=Param(int, 20, msg="number of bootstrap samples to generate"),
                          step=Param(int, 1000, msg="stepsize used to calculate bmus for a pre-trained SOM on testing data"))
    outputs = [('output', QPHandle),
               ('single_NZ', QPHandle),
               ('cellid_output', Hdf5Handle),
               ('uncovered_cell_file', TableHandle)]

    def __init__(self, args, comm=None):
        self.zgrid = None
        self.model = None
        self.usecols = None
        SZPZSummarizer.__init__(self, args, comm=comm)

    def open_model(self, **kwargs):
        SZPZSummarizer.open_model(self, **kwargs)
        self.som = self.model['som']
        self.usecols = self.model['usecols']
        self.column_usage = self.model['column_usage']
        self.ref_column_name = self.model['ref_column']
        self.n_rows = self.model['n_rows']
        self.n_columns = self.model['n_columns']

    def read_photometry_data(self):
        if self.config.hdf5_groupname:
            test_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            test_data = self.get_data('input')

    for col in self.usecols:
        if col not in test_data.keys():  # pragma: no cover
            raise ValueError(f"data column {col} not found in test_data")

    return test_data

    def get_photometry_size(self):
        handle = self.get_handle("input")
        group = self.config.spec_groupname or "/"
        col = self.usecols[0]
        print("should close something here? ", handle)
        return handle.fileObj[group][col].size

    
    def read_spectroscopic_data(self):        
        if self.config.spec_groupname:
            spec_data = self.get_data('spec_input')[self.config.spec_groupname]
        else:  # pragma: no cover
            spec_data = self.get_data('spec_input')

        if self.config.redshift_colname not in spec_data.keys():  # pragma: no cover
            raise ValueError(f"redshift column {self.config.redshift_colname} not found in spec_data")

        return spec_data

    def replace_non_detections(self, data):
        for col in self.usecols:
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                mask = np.isnan(dset[col])
            else:
                mask = np.isclose(dset[col], self.config.nondetect_val)
            dset[col][mask] = self.config.mag_limits[col]

    def set_weight_column(self, data, weight_col):
        if self.config.phot_weightcol == "":
            data["weight"] = np.ones(len(data[self.usecols[0]]))
        elif self.config.phot_weightcol in data.keys():  # pragma: no cover
            data["weight"] = np.array(data[self.config.phot_weightcol])
        # tested in example notebook, so just put a pragma no cover for if present
        else:  # pragma: no cover
            raise KeyError(f"Weight column {weight_col} not present in data!")

    def get_som_coordinates(self, data):
        self.replace_non_detections(data)
        self.set_weight_column(self, data, self.config.phot_weightcol)

        # find the best cells for this data set
        colors = _computemagcolordata(data, self.ref_column_name,
                                           self.usecols, self.column_usage)

        som_coords = get_bmus(self.som, colors, self.config.step).T

        return som_coords

    def save_coordinates(self, start, end, data, som_coords):
        # save this chunk of SOM coordinate information
        coord0, coord1 = som_coords
        ravel_coord = np.ravel_multi_index(cell_ids, (self.n_columns, self.n_rows))

        output = {
            "coord0": coord0,
            "coord1": coord1,
            "ravel_coord": ravel_coord,
        }

        # Leave out for now
        # id_name = self.config.objid_name
        # if id_name and (id_name in data.keys()):  # pragma: no cover
        #     output[id_name] = data[id_name]

        self.cell_id_handle.set_data(output, partial=True)
        self.cell_id_handle.write_chunk(start, end)



    def run(self):
        spec_data = self.read_spectroscopic_data()
        self.replace_non_detections(spec_data)
        self.set_weight_column(self, spec_data, self.config.spec_weightcol)
        sz = spec_data[self.config.redshift_colname]

        # make dictionary of ID data to be written out with cell IDs
        # id_dict = {}



        spec_colors = _computemagcolordata(spec_data, self.ref_column_name,
                                           self.usecols, self.column_usage)

        spec_coords_2d = get_bmus(self.som, spec_colors, self.config.step).T
        spec_coords_1d = np.ravel_multi_index(spec_coords_2d, (self.n_columns, self.n_rows))
        spec_hit_pixels = np.unique(spec_coords_1d)
        num_pixels = self.n_columns * self.n_rows
        


        # We will count the total photometric weight in each SOM pixel.
        # We can also get the object count, just for interest
        phot_pixel_count = np.zeros((self.n_columns, self.n_rows), dtype=np.int64)
        phot_pixel_weight = np.zeros((self.n_columns, self.n_rows))

        # Where we save output data
        ngal_phot = self.get_photometry_size()
        self.cell_id_handle = self.add_handle('cellid_output')
        cell_cols = {
            "coord0":((ngal_phot,), 'i4'),
            "coord1":((ngal_phot,), 'i4'),
            "ravel_coord":((ngal_phot,), 'i4'),
        }
        self.cell_id_handle.initialize_write(ngal_phot, communicator = self.comm, **cell_cols)

        for start, end, test_data in self.iterate_photometric_data():
            print(f"Process {self.rank} adding data {start:,} - {end:,} to SOM")

            phot_som_coords = self.get_som_coordinates(test_data)
            self.save_coordinates(self, start, end, phot_som_coords)

            # acumulate both weight and count in each pixel
            # using np.add.at like this instead of just array[indices] +=1
            # makes it work when the same index appears multiple times
            np.add.at(phot_pixel_counts, phot_som_coords, 1)
            np.add.at(phot_pixel_weight, phot_som_coords, test_data['weight'])

        # If we are running in parallel then we need to sum
        # the results from all the processes
        if self.comm is not None:
            import mpi4py.MPI
            # get the total photometric weight and count in each
            # pixel across all processes and chunks of data
            self.comm.Reduce(mpi4py.MPI.IN_PLACE, phot_pixel_counts)
            self.comm.Reduce(mpi4py.MPI.IN_PLACE, phot_pixel_weight)

            # Only process 0 does the rest of this
            if self.rank != 0:
                return

        # The accumulated weight per pixel, flattened out
        phot_pixel_weight_1d = phot_pixel_weight.flatten()

        # Figure out and report on which pixels were uncovered
        # by the spectroscopy
        phot_missed_pixels = np.where(phot_pixel_weight_1d==0)
        phot_hit_pixels = np.where(phot_pixel_weight_1d!=0)
        uncovered_pixels = np.setdiff1d(phot_hit_pixels, spec_hit_pixels)
        nbad = len(uncovered_pixels)
        print(f"The following {nbad} pixels contain photometric data but not spectroscopic data:")
        print(uncovered_pixels)
        useful_pixels = np.setdiff1d(phot_hit_pixels - uncovered_pixels)
        print(f"{len(useful_pixels)} out of {num_pixels} have usable data")

        # The x and count values of the histogram.  We have a different histogram
        # for each realization, and accumulate each below
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins + 1)
        hist_vals = np.zeros((self.config.nsamples, len(self.zgrid) - 1))

        # Bootstrap over the spectroscopic sample
        rng = np.random.default_rng(seed=self.config.seed)
        ngal_spec = len(spec_coords_1d)


        for i in range(self.config.nsamples):
            # Bootstrap indices and associated weights and redshifts
            bootstrap_indices = rng.integers(low=0, high=ngal_spec, size=ngal_spec)
            bs_specz = sz[bootstrap_indices]
            bs_weights = sweight[bootstrap_indices]
            bs_specz_coords = spec_coords_1d[bootstrap_indices]
            
            # Find spectroscopic galaxies in each pixel and get the histogram
            # of their redshifts. Scaling this by the photometric weight in that
            # pixel gives the redshift distribution for this pixel. Summing that
            # over all the pixels gives the total redshift distribution for this pixel.
            for pix in useful_pixels:
                smask = (bs_specz_coords == pix)
                pix_hist_vals, _ = np.histogram(bs_specz[smask], bins=self.zgrid, weights=bs_weights[smask])
                hist_vals[i] += pix_hist_vals * phot_pixel_weight_1d[pix]

        # Output of the separate bootstrap PDFs, as well as the fiducial one which is 
        # just the mean of all the others
        sample_ens = qp.Ensemble(qp.hist, data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_vals)))
        fid_hist = np.mean(hist_vals, axis=0)
        qp_d = qp.Ensemble(qp.hist, data=dict(bins=self.zgrid, pdfs=fid_hist))
        self.add_data('output', sample_ens)
        self.add_data('single_NZ', qp_d)
        self.add_data('uncovered_cell_file', {"uncovered_pixels": uncovered_pixels})
