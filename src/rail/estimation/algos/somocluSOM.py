import numpy as np
from somoclu import Somoclu
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatInformer
from rail.estimation.summarizer import SZPZSummarizer
from rail.core.data import QPHandle, TableHandle
import qp

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sklearn.cluster as sc
from scipy.spatial.distance import cdist

def_bands = ['u', 'g', 'r', 'i', 'z', 'y']
def_cols = [f"mag_{band}_lsst" for band in def_bands]
def_maglims = dict(mag_u_lsst=27.79,
                   mag_g_lsst=29.04,
                   mag_r_lsst=29.06,
                   mag_i_lsst=28.62,
                   mag_z_lsst=27.98,
                   mag_y_lsst=27.05)


def _computemagcolordata(data, ref_column_name, column_names, colusage):
    if colusage not in ['colors', 'magandcolors', 'columns']:  # pragma: no cover
        raise ValueError(f"column usage value {colusage} is not valid, valid values are 'colors', 'magandcolors', and 'columns'")
    numcols = len(column_names)
    if colusage == 'magandcolors':
        coldata = np.array(data[ref_column_name])
        for i in range(numcols - 1):
            tmpcolor = data[column_names[i]] - data[column_names[i + 1]]
            coldata = np.vstack((coldata, tmpcolor))
    if colusage == 'colors':
        coldata = np.array(data[column_names[0]] - data[column_names[1]])
        for i in range(numcols - 2):
            tmpcolor = data[column_names[i + 1]] - data[column_names[i + 2]]
            coldata = np.vstack((coldata, tmpcolor))
    if colusage == 'columns':  # pragma: no cover
        coldata = np.array(data[column_names[0]])
        for i in range(numcols - 1):
            coldata = np.vstack((coldata, np.array(data[column_names[i + 1]])))
    return coldata.T


def get_surface_data(som, data):  # pragma: no cover
    """Return the Euclidean distance between codebook and data. 
    This is a faster version than the orginal get_surface_data in a somoclu class.
    som: a pre-trained Somoclu object;
    data: np.ndarray of the data vector. If None, then use the training data stored in the som object;
    split: an integer specifying the size of data chunks when calculating the distances.
    returns: The the dot product of the codebook and the data.
    rtype: 2D numpy.array
    """

    d = data

    codebookReshaped = som.codebook.reshape(
        som.codebook.shape[0] * som.codebook.shape[1], som.codebook.shape[2])
    parts = np.array_split(d, 200, axis=0)
    am = np.zeros((data.shape[0], som._n_columns * som._n_rows))

    i = 0
    for part in parts:
        am[i:i+part.shape[0]] = cdist((part), codebookReshaped, 'euclidean')
        i = i+part.shape[0]
    return am


def get_bmus(som, data=None):  # pragma: no cover
    '''
    This function gets the "best matching unit (bmu)" of a given data on a pre-trained SOM.
    It works by multiprocessing chunks of the data.
    Input:
    som: a pre-trained Somoclu object;
    data: np.ndarray of the data vector. If None, then use the training data stored in the som object;
    '''

    if data is None:
        bmus = som.bmus
    else:
        dmap = get_surface_data(som, data)
        bmus = som.get_bmus(dmap)
    return bmus


###

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
        vmin = np.quantile(som_map[~np.isnan(som_map)], 0.01)
        vmax = np.quantile(som_map[~np.isnan(som_map)], 0.99)
    cscale = (som_map - vmin) / (vmax - vmin)
    som_dim = cscale.shape[0]
    if grid_type == 'rectangular':
        ax.matshow(som_map.T, cmap=colormap,
                   vmin=vmin,
                   vmax=vmax)
    else:
        yy, xx = np.meshgrid(np.arange(som_dim), np.arange(som_dim))
        shift = np.zeros(som_dim)
        shift[::2]= -0.5
        xx = xx + shift
        for i in range(cscale.shape[0]):
            for j in range(cscale.shape[1]):
                wy = yy[(i, j)] * np.sqrt(3) / 2
                if np.isnan(cscale[i, j]):
                    color = 'k'
                else:
                    color = colormap(cscale[i, j])

                hex = RegularPolygon((xx[(i, j)], wy),
                                     numVertices=6,
                                     radius=1 / np.sqrt(3),
                                     facecolor=color,
                                     edgecolor=color,
                                     # alpha=.4,
                                     lw=0.2,)
                ax.add_patch(hex)

    scmap = plt.scatter([0, 0], [0, 0], s=0, c=[vmin, vmax],
                        cmap=colormap)
    ax.set_xlim(-1, som_dim - .5)
    ax.set_ylim(-0.5, som_dim * np.sqrt(3) / 2)

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
    """Quick implementation of a SOM-based summarizer. It will
    group a pre-trained SOM into hierarchical clusters and assign
    a galaxy sample into SOM cells and clusters. Then it
    constructs an N(z) estimation via a weighted sum of the
    empirical N(z) consisting of the normalized histogram
    of spec-z values contained in the same SOM cluster as
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
    n_clusters: int
      number of hierarchical clusters of the SOM cells. If not given, the SOM will not be grouped into clusters (or equivalently n_cluster=the total number of SOM cells.)
    Returns
    -------
    qp_ens: qp Ensemble
      ensemble of bootstrap realizations of the estimated N(z) for the input photometric data
    """
    name = 'somocluSOMSummarizer'
    config_options = SZPZSummarizer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          n_clusters=Param(int, -1, msg="The number of hierarchical clusters of SOM cells. If not provided, the SOM cells will not be clustered."),
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
                          nsamples=Param(int, 20, msg="number of bootstrap samples to generate"),)
    outputs = [('output', QPHandle),
               ('single_NZ', QPHandle),
               ('cellid_output', TableHandle),
               ('uncovered_cluster_file', TableHandle)]

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

    def run(self):
        rng = np.random.default_rng(seed=self.config.seed)
        if self.config.hdf5_groupname:
            test_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            test_data = self.get_data('input')
        if self.config.spec_groupname:
            spec_data = self.get_data('spec_input')[self.config.spec_groupname]
        else:  # pragma: no cover
            spec_data = self.get_data('spec_input')
        if self.config.redshift_colname not in spec_data.keys():  # pragma: no cover
            raise ValueError(f"redshift column {self.config.redshift_colname} not found in spec_data")
        sz = spec_data[self.config.redshift_colname]
        for col in self.usecols:
            if col not in test_data.keys():  # pragma: no cover
                raise ValueError(f"data column {col} not found in test_data")

        # make dictionary of ID data to be written out with cell IDs
        id_dict = {}
        if self.config.objid_name != "":  # pragma: no cover
            if self.config.objid_name in test_data.keys():
                id_dict[self.config.objid_name] = test_data[self.config.objid_name]

        # replace nondetects
        dsets = [test_data, spec_data]
        for col in self.usecols:
            for dset in dsets:
                if np.isnan(self.config.nondetect_val):  # pragma: no cover
                    mask = np.isnan(dset[col])
                else:
                    mask = np.isclose(dset[col], self.config.nondetect_val)
                dset[col][mask] = self.config.mag_limits[col]

        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins + 1)
        # assign weight vecs if present, else set all to 1.0
        # tested in example notebook, so just put a pragma no cover for if present
        if self.config.phot_weightcol == "":
            pweight = np.ones(len(test_data[self.usecols[0]]))
        elif self.config.phot_weightcol in test_data.keys():  # pragma: no cover
            pweight = np.array(test_data[self.config.phot_weightcol])
        else:  # pragma: no cover
            raise KeyError(f"photometric weight column {self.config.phot_weightcol} not present in data!")
        if self.config.spec_weightcol == "":
            sweight = np.ones(len(spec_data[self.usecols[0]]))
        elif self.config.spec_weightcol in test_data.keys():  # pragma: no cover
            sweight = np.array(spec_data[self.config.spec_weightcol])
        else:  # pragma: no cover
            raise KeyError(f"spectroscopic weight column {self.config.spec_weightcol} not present in data!")

        # find the best cells for the photometric and spectrosopic datasets
        phot_colors = _computemagcolordata(test_data, self.ref_column_name,
                                           self.usecols, self.column_usage)
        spec_colors = _computemagcolordata(spec_data, self.ref_column_name,
                                           self.usecols, self.column_usage)

        if self.config.n_clusters > self.n_rows * self.n_columns:  # pragma: no cover
            print("Warning: number of clusters cannot be greater than the number of cells ("+str(self.n_rows * self.n_columns)+"). The SOM will NOT be grouped into clusters.")
            n_clusters = self.n_rows * self.n_columns
        elif self.config.n_clusters == -1:
            print("Warning: number of clusters is not provided. The SOM will NOT be grouped into clusters.")
            n_clusters = self.n_rows * self.n_columns
        else:  # pragma: no cover
            n_clusters = self.config.n_clusters

        algorithm = sc.AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        self.som.cluster(algorithm)
        som_cluster_inds = self.som.clusters.reshape(-1)

        phot_som_coords = get_bmus(self.som, phot_colors).T
        spec_som_coords = get_bmus(self.som, spec_colors).T
        phot_pixel_coords = np.ravel_multi_index(phot_som_coords, (self.n_columns, self.n_rows))
        spec_pixel_coords = np.ravel_multi_index(spec_som_coords, (self.n_columns, self.n_rows))

        phot_som_clusterind = som_cluster_inds[phot_pixel_coords]
        spec_som_clusterind = som_cluster_inds[spec_pixel_coords]

        # add id coords to id_dict for writeout
        xcoord, ycoord = phot_som_coords
        id_dict['coord0'] = xcoord
        id_dict['coord1'] = ycoord
        id_dict['ravel_coord'] = phot_pixel_coords
        id_dict['cluster_ind'] = phot_som_clusterind
        id_dict['cell_ravel_ind'] = phot_pixel_coords

        ngal = len(spec_pixel_coords)
        phot_cluster_set = set(phot_som_clusterind)
        spec_cluster_set = set(spec_som_clusterind)
        uncovered_clusters = phot_cluster_set - spec_cluster_set
        bad_cluster = dict(uncovered_clusters=np.array(list(uncovered_clusters)))
        print("the following clusters contain photometric data but not spectroscopic data:")
        print(uncovered_clusters)
        useful_clusters = phot_cluster_set - uncovered_clusters
        print(f"{len(useful_clusters)} out of {n_clusters} have usable data")

        # effective number defined in Heymans et al. (2012) to quantify the photometric representation.
        # also see Eq.7 in Wright et al. (2020).
        # Note that the origional definition should be effective number *density*, which equals to N_eff / Area.
        N_eff = np.sum(pweight) ** 2 / np.sum(pweight**2)

        hist_vals = np.empty((self.config.nsamples, len(self.zgrid) - 1))

        N_eff_p_samples = np.zeros(self.config.nsamples)
        for i in range(self.config.nsamples):
            bootstrap_indices = rng.integers(low=0, high=ngal, size=ngal)
            bs_specz = sz[bootstrap_indices]
            bs_weights = sweight[bootstrap_indices]
            bs_specz_clusters = spec_som_clusterind[bootstrap_indices]
            tmp_hist_vals = np.zeros(len(self.zgrid) - 1)
            n_eff_p_num = 0
            n_eff_p_den = 0
            for cluster in useful_clusters:
                pmask = (phot_som_clusterind == cluster)
                binpweight = np.sum(pweight[pmask])
                smask = (bs_specz_clusters == cluster)
                cluster_hist_vals, _ = np.histogram(bs_specz[smask], bins=self.zgrid, weights=bs_weights[smask])
                tmp_hist_vals += cluster_hist_vals * binpweight

                n_eff_p_num += np.sum(pweight[pmask])
                n_eff_p_den += np.sum(pweight[pmask] ** 2)
            N_eff_p_samples[i] = n_eff_p_num ** 2 / n_eff_p_den
            hist_vals[i, :] = tmp_hist_vals

        # the effective number density of the subsample of the photometric sample reside within SOM groupings which contain spectroscopy
        N_eff_p = np.mean(N_eff_p_samples)

        # the ratio between the effective number of photometric sub-sample that has spectroscopic representation and the full photometric sample.
        # We use this to evaluate the spectroscopic representation of current SOM setup and calibrating spectroscopic catalog.
        self.neff_p_to_neff = N_eff_p / N_eff
        sample_ens = qp.Ensemble(qp.hist, data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_vals)))
        fid_hist = np.mean(hist_vals, axis=0)
        qp_d = qp.Ensemble(qp.hist, data=dict(bins=self.zgrid, pdfs=fid_hist))
        self.add_data('output', sample_ens)
        self.add_data('single_NZ', qp_d)
        self.add_data('uncovered_cluster_file', bad_cluster)
        self.add_data('cellid_output', id_dict)
