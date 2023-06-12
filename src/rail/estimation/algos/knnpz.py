"""
quick implementation of k nearest neighbor estimator
First pass will ignore photometric errors and just do
things in terms of magnitudes, we will expand in a
future update
"""

import numpy as np
import copy

from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer

from rail.evaluation.metrics.cdeloss import CDELoss
from rail.core.common_params import SHARED_PARAMS

import pandas as pd
import qp



def _computecolordata(df, ref_column_name, column_names):
    newdict = {}
    newdict['x'] = df[ref_column_name]
    nbands = len(column_names) - 1
    for k in range(nbands):
        newdict[f'x{k}'] = df[column_names[k]] - df[column_names[k + 1]]
    newdf = pd.DataFrame(newdict)
    coldata = newdf.to_numpy()
    return coldata


def _makepdf(dists, ids, szs, sigma):
    sigmas = np.full_like(dists, sigma)
    weights = 1. / dists
    weights /= weights.sum(axis=1, keepdims=True)
    means = szs[ids]
    pdfs = qp.Ensemble(qp.mixmod, data=dict(means=means, stds=sigmas, weights=weights))
    return pdfs


class Inform_KNearNeighPDF(CatInformer):
    """Train a KNN-based estimator
    """
    name = 'Inform_KNearNeighPDF'
    config_options = CatInformer.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          hdf5_groupname=SHARED_PARAMS,
                          trainfrac=Param(float, 0.75,
                                          msg="fraction of training data used to make tree, rest used to set best sigma"),
                          seed=Param(int, 0, msg="Random number seed for NN training"),
                          sigma_grid_min=Param(float, 0.01, msg="minimum value of sigma for grid check"),
                          sigma_grid_max=Param(float, 0.075, msg="maximum value of sigma for grid check"),
                          ngrid_sigma=Param(int, 10, msg="number of grid points in sigma check"),
                          leaf_size=Param(int, 15, msg="min leaf size for KDTree"),
                          nneigh_min=Param(int, 3, msg="int, min number of near neighbors to use for PDF fit"),
                          nneigh_max=Param(int, 7, msg="int, max number of near neighbors to use ofr PDF fit"))

    def __init__(self, args, comm=None):
        """ Constructor
        Do CatInformer specific initialization, then check on bands """
        CatInformer.__init__(self, args, comm=comm)

        usecols = self.config.bands.copy()
        usecols.append(self.config.redshift_col)
        self.usecols = usecols
        self.zgrid = None

    def run(self):
        """
        train a KDTree on a fraction of the training data
        """
        from sklearn.neighbors import KDTree
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data('input')
        knndf = pd.DataFrame(training_data, columns=self.config.bands)
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)

        # replace nondetects
        # will fancy this up later with a flow to sample from truth
        for col in self.config.bands:
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                knndf.loc[np.isnan(knndf[col]), col] = self.config.mag_limits[col]
            else:
                knndf.loc[np.isclose(knndf[col], self.config.nondetect_val), col] = self.config.mag_limits[col]

        trainszs = np.array(training_data[self.config.redshift_col])
        colordata = _computecolordata(knndf, self.config.ref_band, self.config.bands)
        nobs = colordata.shape[0]
        rng = np.random.default_rng(seed=self.config.seed)
        perm = rng.permutation(nobs)
        ntrain = round(nobs * self.config.trainfrac)
        xtrain_data = colordata[perm[:ntrain]]
        train_data = copy.deepcopy(xtrain_data)
        val_data = colordata[perm[ntrain:]]
        xtrain_sz = trainszs[perm[:ntrain]].copy()
        train_sz = np.array(copy.deepcopy(xtrain_sz))
        val_sz = np.array(trainszs[perm[ntrain:]])
        print(f"split into {len(train_sz)} training and {len(val_sz)} validation samples")
        tmpmodel = KDTree(train_data, leaf_size=self.config.leaf_size)
        # Find best sigma and n_neigh by minimizing CDE Loss
        bestloss = 1e20
        bestsig = self.config.sigma_grid_min
        bestnn = self.config.nneigh_min
        siggrid = np.linspace(self.config.sigma_grid_min, self.config.sigma_grid_max, self.config.ngrid_sigma)
        print("finding best fit sigma and NNeigh...")
        for sig in siggrid:
            for nn in range(self.config.nneigh_min, self.config.nneigh_max + 1):
                dists, idxs = tmpmodel.query(val_data, k=nn)
                ens = _makepdf(dists, idxs, train_sz, sig)
                cdelossobj = CDELoss(ens, self.zgrid, val_sz)
                cdeloss = cdelossobj.evaluate().statistic
                if cdeloss < bestloss:
                    bestsig = sig
                    bestnn = nn
                    bestloss = cdeloss
        numneigh = bestnn
        sigma = bestsig
        print(f"\n\n\nbest fit values are sigma={sigma} and numneigh={numneigh}\n\n\n")
        # remake tree with full dataset!
        kdtree = KDTree(colordata, leaf_size=self.config.leaf_size)
        self.model = dict(kdtree=kdtree, bestsig=sigma, nneigh=numneigh, truezs=trainszs)
        self.add_data('model', self.model)


class KNearNeighPDF(CatEstimator):
    """KNN-based estimator
    """
    name = 'KNearNeighPDF'
    config_options = CatEstimator.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS)

    def __init__(self, args, comm=None):
        """ Constructor:
        Do Estimator specific initialization """
        self.sigma = None
        self.numneigh = None
        self.model = None
        self.trainszs = None
        self.zgrid = None
        CatEstimator.__init__(self, args, comm=comm)
        usecols = self.config.bands.copy()
        usecols.append(self.config.redshift_col)
        self.usecols = usecols

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.model is None:  #pragma: no cover
            return
        self.sigma = self.model['bestsig']
        self.numneigh = self.model['nneigh']
        self.kdtree = self.model['kdtree']
        self.trainszs = self.model['truezs']

    def _process_chunk(self, start, end, data, first):
        """
        calculate and return PDFs for each galaxy using the trained flow
        """
        print(f"Process {self.rank} estimating PZ PDF for rows {start:,} - {end:,}")
        knn_df = pd.DataFrame(data, columns=self.config.bands)
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)

        # replace nondetects
        # will fancy this up later with a flow to sample from truth
        for col in self.config.bands:
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                knn_df.loc[np.isnan(knn_df[col]), col] = self.config.mag_limits[col]
            else:
                knn_df.loc[np.isclose(knn_df[col], self.config.nondetect_val), col] = self.config.mag_limits[col]

        testcolordata = _computecolordata(knn_df, self.config.ref_band, self.config.bands)
        dists, idxs = self.kdtree.query(testcolordata, k=self.numneigh)
        test_ens = _makepdf(dists, idxs, self.trainszs, self.sigma)
        zmode = test_ens.mode(grid=self.zgrid)
        test_ens.set_ancil(dict(zmode=zmode))
        self._do_chunk_output(test_ens, start, end, first)
