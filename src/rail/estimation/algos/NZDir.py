"""
Implement simple version of TxPipe NZDir summarizer
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.data import QPHandle
import qp
import scipy.spatial
import pandas as pd


class Inform_NZDir(CatInformer):
    """Quick implementation of an NZ Estimator that
    creates weights for each input
    object using sklearn's NearestNeighbors.
    Very basic, we can probably
    create a more sophisticated SOM-based DIR method in
    the future.
    This inform stage just creates a nearneigh model
    of the spec-z data and some distances to N-th
    neighbor that will be used in the estimate stage.

    Notes
    -----
    This will create `model` a dictionary of the nearest neighboor model and params used by estimate

    """
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    default_usecols = [f"mag_{band}_lsst" for band in bands]

    name = 'Inform_NZDir'
    config_options = CatInformer.config_options.copy()
    config_options.update(usecols=Param(list, default_usecols, msg="columns from sz_data for Neighor calculation"),
                          n_neigh=Param(int, 10, msg="number of neighbors to use"),
                          kalgo=Param(str, "kd_tree", msg="Neighbor algorithm to use"),
                          kmetric=Param(str, "euclidean", msg="Knn metric to use"),
                          szname=Param(str, "redshift", msg="name of specz column in sz_data"),
                          szweightcol=Param(str, "", msg="name of sz weight column"),
                          distance_delta=Param(float, 1.e-6, msg="padding for distance calculation"),
                          hdf5_groupname=Param(str, "photometry", msg="name of hdf5 group for data, if None, then set to ''"))

    def __init__(self, args, comm=None):
        """ Constructor:
        Do Informer specific initialization """
        CatInformer.__init__(self, args, comm=comm)

    def run(self):
        from sklearn.neighbors import NearestNeighbors

        if self.config.hdf5_groupname:
            sz_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            sz_data = self.get_data('input')
        # if no weights specified, which is the default
        # set all weights to 1.0 if weights not present in keys
        # set to same length as one of the usecols cols
        if self.config.szweightcol == '':
            szweights = np.ones(len(sz_data[self.config.usecols[0]]))
        elif self.config.szweightcol in sz_data.keys():  # pragma: no cover
            szweights = np.array(sz_data[self.config.szweightcol])
        else:  # pragma: no cover
            raise KeyError(f"weight column {self.config.szweightcol} not found in input data!")
        sz_mag_data = np.array([sz_data[band] for band in self.config.usecols]).T
        sz_mag_data[~np.isfinite(sz_mag_data)] = 40.
        szvec = np.array(sz_data[self.config.szname])
        neighbors = NearestNeighbors(
            n_neighbors=self.config["n_neigh"],
            algorithm=self.config["kalgo"],
            metric=self.config["kmetric"],
        ).fit(sz_mag_data)
        # find distance to Kth nearest neighbor, will be used later to determine how far
        # away to search a second tree of photo data
        distances, _ = neighbors.kneighbors(sz_mag_data)
        distances = np.array(np.amax(distances, axis=1) + self.config["distance_delta"])
        self.model = dict(distances=distances,
                          szusecols=self.config.usecols, szweights=szweights,
                          szvec=szvec, sz_mag_data=sz_mag_data)
        self.add_data('model', self.model)


class NZDir(CatEstimator):
    """Quick implementation of a summarizer that creates
    weights for each input object using sklearn's
    NearestNeighbors.  Very basic, we can probably
    create a more sophisticated SOM-based DIR method in
    the future
    Parameters
    ----------
    zmin: float
        min redshift for z grid
    zmax: float
        max redshift for z grid
    nzbins: int
        number of bins in z grid

    Returns
    -------
    qp_ens: qp Ensemble
        histogram Ensemble describing N(z) estimate
    """

    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    default_usecols = [f"mag_{band}_lsst" for band in bands]

    name = 'NZDir'
    config_options = CatEstimator.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          seed=Param(int, 87, msg="random seed"),
                          usecols=Param(list, default_usecols, msg="columns from sz_date for Neighor calculation"),
                          leafsize=Param(int, 40, msg="leaf size for testdata KDTree"),
                          hdf5_groupname=Param(str, "photometry", msg="name of hdf5 group for data, if None, then set to ''"),
                          phot_weightcol=Param(str, "", msg="name of photometry weight, if present"),
                          nsamples=Param(int, 20, msg="number of bootstrap samples to generate"))
    outputs = [('output', QPHandle),
               ('single_NZ', QPHandle)]

    def __init__(self, args, comm=None):
        self.zgrid = None
        self.model = None
        self.distances = None
        self.szusecols = None
        self.szweights = None
        self.sz_mag_data = None
        self.bincents = None
        CatEstimator.__init__(self, args, comm=comm)

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        self.distances = self.model['distances']
        self.szusecols = self.model['szusecols']
        self.szweights = self.model['szweights']
        self.szvec = self.model['szvec']
        self.sz_mag_data = self.model['sz_mag_data']

    def run(self):
        rng = np.random.default_rng(seed=self.config.seed)
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins + 1)
        self.bincents = 0.5 * (self.zgrid[1:] + self.zgrid[:-1])

        # Creating de indices for the bootstrap sampling, and broadcasting to the other process
        # if we are running in parallel
        ngal = len(self.szweights)
        nsamp = self.config.nsamples
        if self.rank == 0:
            bootstrap_matrix = rng.integers(low=0, high=ngal, size=(ngal,nsamp))
        else:  # pragma: no cover
            bootstrap_matrix = None

        if self.comm is not None:  # pragma: no cover
            bootstrap_matrix = self.comm.bcast(bootstrap_matrix, root = 0)

        iterator = self.input_iterator('input')
        first = True
        self._initialize_run()
        self._output_handle = None
        total_chunks = int(np.ceil(self._input_length/self.config.chunk_size))
        for s, e, test_data in iterator:
            print(f"Process {self.rank} running estimator on chunk {s} - {e}")

            total_chunks = int(np.ceil(self._input_length/self.config.chunk_size))
            chunk_number = s//self.config.chunk_size
            self._process_chunk(first, total_chunks, chunk_number, test_data, bootstrap_matrix)
            first = False
        self._single_handle.finalize_write()
        self._sample_handle.finalize_write()
        if self.comm is not None:  # pragma: no cover
            self.comm.Barrier()
        if self.rank == 0:
            # Joining the histograms and updating the data handles
            self.join_histograms()




    def _process_chunk(self, first, number_of_chunks, chunk_number, test_data, bootstrap_matrix):
        # assign weight vecs if present, else set all to 1.0
        # tested in example notebook, so just put a pragma no cover for if present
        if self.config.phot_weightcol == "":
            pweight = np.ones(len(test_data[self.config.usecols[0]]))
        elif self.config.phot_weightcol in test_data.keys():  # pragma: no cover
            pweight = np.array(test_data[self.config.phot_weightcol])
        else:
            raise KeyError(f"photometric weight column {self.config.phot_weightcol} not present in data!")
        # calculate weights for the test data
        weights = np.zeros(len(self.szvec))
        tmpdf = pd.DataFrame(test_data)

        phot_mag_data = np.array([tmpdf[band] for band in self.config.usecols]).T
        phot_mag_data[~np.isfinite(phot_mag_data)] = 40.
        # create a tree for the photometric data, for each specz object find all the
        # tomo objects within the distance to Kth speczNN from before
        tree = scipy.spatial.KDTree(phot_mag_data, leafsize=self.config.leafsize)
        indices = tree.query_ball_point(self.sz_mag_data, self.distances)

        # for each of the indexed galaxies within the distance, add the weights to the
        # appropriate tomographic bin
        for j, index in enumerate(indices):
            weights[j] += pweight[index].sum()
        # make weighted histograms
        hist_data = np.histogram(
            self.szvec,
            bins=self.zgrid,
            weights=weights * self.szweights,
        )
        normalization_factor = hist_data[0].sum()
        qp_dstn = qp.Ensemble(qp.hist,
                   data=dict(bins=self.zgrid, pdfs=hist_data[0]),ancil=dict(normalization=np.array([normalization_factor])))
        if first:
            self._single_handle = self.initialize_handle('single_NZ', qp_dstn, number_of_chunks)
        self._do_chunk_output(qp_dstn, chunk_number, chunk_number+1, self._single_handle)

        # The way things are set up, it is easier and faster to bootstrap the spec-z gals
        # and weights, but if we wanted to be more like the other bootstraps we should really
        # bootstrap the photometric data and re-run the ball tree query N times.
        nsamp = self.config.nsamples
        hist_vals = np.empty((nsamp, self.config.nzbins))
        for i in range(nsamp):
            bootstrap_indices = bootstrap_matrix[:,i]
            zarr = self.szvec[bootstrap_indices]
            tmpweight = self.szweights[bootstrap_indices] * weights[bootstrap_indices]
            tmp_hist_vals = np.histogram(zarr, bins=self.zgrid, weights=tmpweight)[0]
            hist_vals[i] = tmp_hist_vals
        normalization_factor = hist_vals.sum(axis=(1,))
        sample_ens = qp.Ensemble(qp.hist,
                         data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_vals)),ancil=dict(normalization=normalization_factor))
        npdf = nsamp*number_of_chunks
        if first:
            self._sample_handle = self.initialize_handle('output', sample_ens, npdf)
        self._do_chunk_output(sample_ens, chunk_number*nsamp, (chunk_number+1)*nsamp, self._sample_handle)


    def initialize_handle(self, tag, data, npdf):
            handle = self.add_handle(tag, data = data)
            handle.initialize_write(npdf, communicator = self.comm)
            return handle

    def _do_chunk_output(self, qp_dstn, start, end, handle):
        handle.set_data(qp_dstn, partial=True)
        handle.write_chunk(start, end)

    def join_histograms(self):
        qp_d_spread = self._single_handle.read(force=True)
        tab_single = qp_d_spread.build_tables()

        hist_single = np.zeros(self.config.nzbins)
        normalization = tab_single['ancil']['normalization']
        total_chunks = len(normalization)
        for i in range(total_chunks):
            hist_single += tab_single['data']['pdfs'][i]*normalization[i]

        sample_ens_spread = self._sample_handle.read(force=True)
        tab_samples = sample_ens_spread.build_tables()

        nsamp = self.config.nsamples
        hist_samples = np.zeros((nsamp,self.config.nzbins))
        normalization = tab_samples['ancil']['normalization']
        hist_samples_spread = tab_samples['data']['pdfs']*normalization[:,None]
        for i in range(total_chunks):
            hist_samples += hist_samples_spread[i*nsamp:(i+1)*nsamp]

        sample_ens = qp.Ensemble(qp.hist, data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_samples)))
        qp_d = qp.Ensemble(qp.hist, data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_single)))
        self._single_handle.set_data(qp_d)
        self._sample_handle.set_data(sample_ens)
        self._single_handle.write()
        self._sample_handle.write()