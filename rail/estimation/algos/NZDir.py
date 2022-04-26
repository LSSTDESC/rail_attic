"""
Implement simple version of TxPipe NZDir summarizer
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.estimator import Estimator, Informer
import qp
from sklearn.neighbors import NearestNeighbors
import scipy.spatial
import pandas as pd


class Inform_NZDir(Informer):
    """Quick implementation of an NZ Estimator that
    creates weights for each input
    object using sklearn's NearestNeighbors.
    Very basic, we can probably
    create a more sophisticated SOM-based DIR method in
    the future.
    This inform stage just creates a nearneigh model
    of the spec-z data and some distances to N-th
    neighbor that will be used in the estimate stage.
    Returns:
    --------
    model: dict
      dictionary of nearest neigh model and params used by estimate
    """
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    default_usecols = [f"mag_{band}_lsst" for band in bands]

    name = 'Inform_NZDir'
    config_options = Informer.config_options.copy()
    config_options.update(usecols=Param(list, default_usecols, msg="columns from sz_date for Neighor calculation"),
                          n_neigh=Param(int, 10, msg="number of neighbors to use"),
                          kalgo=Param(str, "kd_tree", msg="Neighbor algorithm to use"),
                          kmetric=Param(str, "euclidean", msg="Knn metric to use"),
                          szname=Param(str, "redshift", msg="name of specz column in sz_data"),
                          szweightcol=Param(str, "weight", msg="name of sz weight column"),
                          distance_delta=Param(float, 1.e-6, msg="padding for distance calculation"),
                          hdf5_groupname=Param(str, "photometry", msg="name of hdf5 group for data, if None, then set to ''"))

    def __init__(self, args, comm=None):
        """ Constructor:
        Do Informer specific initialization """
        Informer.__init__(self, args, comm=comm)

    def run(self):
        if self.config.hdf5_groupname:
            sz_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma:  no cover
            sz_data = self.get_data('input')
        if self.config.szweightcol in sz_data.keys():  # pragma: no cover
            szweights = np.array(sz_data[self.config.szweightcol])
        else:
            szweights = np.ones(len(sz_data[self.config.usecols[0]]))
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


class NZDir(Estimator):
    """Quick implementation of a summarizer that creates
    weights for each input object using sklearn's
    NearestNeighbors.  Very basic, we can probably
    create a more sophisticated SOM-based DIR method in
    the future
    Parameters:
    -----------
    zmin: float
      min redshift for z grid
    zmax: float
      max redshift for z grid
    nzbins: int
      number of bins in z grid
    Returns:
    qp_ens: qp Ensemble
      histogram Ensemble describing N(z) estimate
    """

    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    default_usecols = [f"mag_{band}_lsst" for band in bands]

    name = 'NZDir'
    config_options = Estimator.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          usecols=Param(list, default_usecols, msg="columns from sz_date for Neighor calculation"),
                          bincol=Param(str, "bin", msg="name of column with int bins"),
                          leafsize=Param(int, 40, msg="leaf size for testdata KDTree"),
                          hdf5_groupname=Param(str, "photometry", msg="name of hdf5 group for data, if None, then set to ''"),
                          phot_weightcol=Param(str, "weight", msg="name of photometry weight, if present"))

    def __init__(self, args, comm=None):
        self.zgrid = None
        self.model = None
        self.distances = None
        self.szusecols = None
        self.szweights = None
        self.sz_mag_data = None
        Estimator.__init__(self, args, comm=comm)

    def open_model(self, **kwargs):
        Estimator.open_model(self, **kwargs)
        self.distances = self.model['distances']
        self.szusecols = self.model['szusecols']
        self.szweights = self.model['szweights']
        self.szvec = self.model['szvec']
        self.sz_mag_data = self.model['sz_mag_data']

    def run(self):
        if self.config.hdf5_groupname:
            test_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma:  no cover
            test_data = self.get_data('input')
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)

        # assign weight vecs, if present, else set all to 1.0
        # tested in example notebook, so just put a pragma no cover for if present
        if self.config.phot_weightcol in test_data.keys():  # pragma: no cover
            pweight = np.array(test_data[self.config.phot_weightcol])
        else:
            pweight = np.ones(len(test_data[self.config.usecols[0]]))
        # calculate weights for the test data
        if self.config.bincol not in test_data.keys():
            # stack all galaxies into single ensemble
            print("bincol not found! Perform single tomo bin measurement!")
            tomobinids = np.ones(len(test_data[self.config.usecols[0]]), dtype='int')
            test_data[self.config.bincol] = tomobinids
        else:  # pragma: no cover
            print("found bin IDs")
            tomobinids = test_data[self.config.bincol]
            print(tomobinids)
        nzbins = sorted(list(set(tomobinids)))
        print(f" list of bins to process: {nzbins}")
        numnzbins = len(nzbins)
        weights = np.zeros([numnzbins, len(self.szvec)])
        hist_data = []
        tmpdf = pd.DataFrame(test_data)

        for i, bin in enumerate(nzbins):
            print(f" working on tomobin #{i} for bin defined by {bin}")
            binselect = tmpdf[self.config.bincol] == bin
            cutdata = tmpdf[binselect]

            phot_mag_data = np.array([cutdata[band] for band in self.config.usecols]).T
            phot_mag_data[~np.isfinite(phot_mag_data)] = 40.
            # create a tree for the photometric data, for each specz object find all the
            # tomo objects within the distance to Kth speczNN from before
            tree = scipy.spatial.KDTree(phot_mag_data, leafsize=self.config.leafsize)
            indices = tree.query_ball_point(self.sz_mag_data, self.distances)

            # for each of the indexed galaxies within the distance, add the weights to the
            # appropriate tomographic bin
            for j, index in enumerate(indices):
                weights[i, j] += pweight[index].sum()
            # make weighted histograms
            single_hist = np.histogram(
                self.szvec,
                bins=self.zgrid,
                range=(0, self.config["zmax"]),
                weights=weights[i] * self.szweights,
            )
            hist_data.append(single_hist[0])
        # make the ensembles of the histograms
        qp_d = qp.Ensemble(qp.hist,
                           data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_data)))
        self.add_data('output', qp_d)
