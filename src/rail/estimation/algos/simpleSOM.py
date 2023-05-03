import numpy as np
from minisom import MiniSom
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatInformer
from rail.estimation.summarizer import SZPZSummarizer
from rail.core.data import QPHandle, TableHandle
from rail.core.common_params import SHARED_PARAMS

import qp



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


class Inform_SimpleSOMSummarizer(CatInformer):
    """Summarizer that uses a SOM to construct a weighted sum
    of spec-z objects in the same SOM cell as each photometric
    galaxy in order to estimate the overall N(z).  This is
    very related to the NZDir estimator, though that estimator
    actually reverses this process and looks for photometric
    neighbors around each spectroscopic galaxy, which can
    lead to problems if there are photometric galaxies with
    no nearby spec-z objects (NZDir is not aware that such
    objects exist and thus can hid biases).
    Part of the SimpeSOM estimator will be a check for cells
    which contain photometric objects but do not contain any
    corresponding training/spec-z objects, those unmatched
    objects will be flagged for possible removal from the
    input sample.
    The inform stage will simply construct a 2D grid SOM
    using `minisom` from a large sample of input
    photometric data and save this as an output.  This may
    be a computationally intensive stage, though it will
    hopefully be run once and used by the estimate/summarize
    stage many times without needing to be re-run.

    We can make the SOM either with all colors, or one
    magnitude and N colors, or an arbitrary set of columns.
    The code includes a flag `column_usage` to set usage,
    If set to "colors" it will take the difference of each
    adjacen pair of columns in `bands` as the colors. If
    set to `magandcolors` it will use these colors plus one
    magnitude as specified by `ref_band`.  If set to
    `columns` then it will take as inputs all of the columns
    specified by `bands` (they can be magnitudes, colors,
    or any other input specified by the user).  NOTE: any
    custom `bands` parameters must have an accompanying
    `nondetect_val` dictionary that will replace
    nondetections with the nondetect_val values!

    Returns
    -------
    model: pickle file
      pickle file containing the `minisom` SOM object that
    will be used by the estimation/summarization stage
    """
    name = 'Inform_SimpleSOM'
    config_options = CatInformer.config_options.copy()
    config_options.update(nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          hdf5_groupname=SHARED_PARAMS,
                          column_usage=Param(str, "magandcolors", msg="switch for how SOM uses columns, "
                                             + "valid values are 'colors', 'magandcolors', and 'columns'"),
                          seed=Param(int, 0, msg="Random number seed"),
                          m_dim=Param(int, 31, msg="number of cells in SOM y dimension"),
                          n_dim=Param(int, 31, msg="number of cells in SOM x dimension"),
                          som_sigma=Param(float, 1.5, msg="sigma param in SOM training"),
                          som_learning_rate=Param(float, 0.5, msg="SOM learning rate"),
                          som_iterations=Param(int, 10_000, msg="number of iterations in SOM training"))

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
        for col in self.config.bands:
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                mask = np.isnan(training_data[col])
            else:
                mask = np.isclose(training_data[col], self.config.nondetect_val)
            training_data[col][mask] = self.config.mag_limits[col]

        colors = _computemagcolordata(training_data, self.config.ref_band,
                                      self.config.bands, self.config.column_usage)

        som = MiniSom(self.config.n_dim, self.config.m_dim, colors.shape[1],
                      sigma=self.config.som_sigma,
                      learning_rate=self.config.som_learning_rate,
                      neighborhood_function='gaussian',
                      random_seed=self.config.seed)
        som.pca_weights_init(colors)
        som.train(colors, self.config.som_iterations, verbose=True)

        modeldict = dict(som=som, usecols=self.config.bands,
                         ref_column=self.config.ref_band,
                         m_dim=self.config.m_dim, n_dim=self.config.n_dim,
                         column_usage=self.config.column_usage)
        self.model = modeldict
        self.add_data('model', self.model)


class SimpleSOMSummarizer(SZPZSummarizer):
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
    using the n_dim and m_dim parameters.
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
    name = 'SimpleSOMSummarizer'
    config_options = SZPZSummarizer.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          hdf5_groupname=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          objid_name=Param(str, "", "name of ID column, if present will be written to cellid_output"),
                          spec_groupname=Param(str, "photometry", msg="name of hdf5 group for spec data, if None, then set to ''"),
                          seed=Param(int, 12345, msg="random seed"),
                          phot_weightcol=Param(str, "", msg="name of photometry weight, if present"),
                          spec_weightcol=Param(str, "", msg="name of specz weight col, if present"),
                          nsamples=Param(int, 20, msg="number of bootstrap samples to generate"))
    outputs = [('output', QPHandle),
               ('single_NZ', QPHandle),
               ('cellid_output', TableHandle),
               ('uncovered_cell_file', TableHandle)]

    def __init__(self, args, comm=None):
        self.zgrid = None
        self.model = None
        self.usecols = None
        SZPZSummarizer.__init__(self, args, comm=comm)
        self.som = None
        self.column_usage = None
        self.ref_column_name = None
        self.m_dim = None
        self.n_dim = None

    def run(self):
        self.som = self.model['som']
        self.usecols = self.model['usecols']
        self.column_usage = self.model['column_usage']
        self.ref_column_name = self.model['ref_column']
        self.m_dim = self.model['m_dim']
        self.n_dim = self.model['n_dim']        
        rng = np.random.default_rng(seed=self.config.seed)
        if self.config.hdf5_groupname:
            test_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            test_data = self.get_data('input')
        if self.config.spec_groupname:
            spec_data = self.get_data('spec_input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            spec_data = self.get_data('spec_input')
        if self.config.redshift_col not in spec_data.keys():  # pragma: no cover
            raise ValueError(f"redshift column {self.config.redshift_colname} not found in spec_data")
        sz = spec_data[self.config.redshift_col]
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

        phot_som_coords = np.array([self.som.winner(x) for x in phot_colors]).T
        spec_som_coords = np.array([self.som.winner(x) for x in spec_colors]).T
        phot_pixel_coords = np.ravel_multi_index(phot_som_coords, (self.n_dim, self.m_dim))
        spec_pixel_coords = np.ravel_multi_index(spec_som_coords, (self.n_dim, self.m_dim))

        # add id coords to id_dict for writeout
        xcoord, ycoord = phot_som_coords
        id_dict['coord0'] = xcoord
        id_dict['coord1'] = ycoord
        id_dict['ravel_coord'] = phot_pixel_coords

        num_pixels = self.n_dim * self.m_dim
        ngal = len(spec_pixel_coords)
        phot_pixel_set = set(phot_pixel_coords)
        spec_pixel_set = set(spec_pixel_coords)
        uncovered_pixels = phot_pixel_set - spec_pixel_set
        bad_pix = dict(uncovered_pixels=np.array(list(uncovered_pixels)))
        print("the following pixels contain photometric data but not spectroscopic data:")
        print(uncovered_pixels)
        useful_pixels = phot_pixel_set - uncovered_pixels
        print(f"{len(useful_pixels)} out of {num_pixels} have usable data")

        hist_vals = np.empty((self.config.nsamples, len(self.zgrid) - 1))
        for i in range(self.config.nsamples):
            bootstrap_indices = rng.integers(low=0, high=ngal, size=ngal)
            bs_specz = sz[bootstrap_indices]
            bs_weights = sweight[bootstrap_indices]
            bs_specz_coords = spec_pixel_coords[bootstrap_indices]
            tmp_hist_vals = np.zeros(len(self.zgrid) - 1)
            for pix in useful_pixels:
                pmask = (phot_pixel_coords == pix)
                binpweight = np.sum(pweight[pmask])
                smask = (bs_specz_coords == pix)
                pix_hist_vals, _ = np.histogram(bs_specz[smask], bins=self.zgrid, weights=bs_weights[smask])
                tmp_hist_vals += pix_hist_vals * binpweight
            hist_vals[i, :] = tmp_hist_vals

        sample_ens = qp.Ensemble(qp.hist, data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_vals)))
        fid_hist = np.mean(hist_vals, axis=0)
        qp_d = qp.Ensemble(qp.hist, data=dict(bins=self.zgrid, pdfs=fid_hist))
        self.add_data('output', sample_ens)
        self.add_data('single_NZ', qp_d)
        self.add_data('uncovered_cell_file', bad_pix)
        self.add_data('cellid_output', id_dict)
